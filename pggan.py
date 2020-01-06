import torch
import torch.nn as nn
import torch.optim as optim

import layers as ll
import losses

class PGGAN(nn.Module):
    def __init__(self, args, device='cuda'):
        super().__init__()
        self.args = args
        self.device = device
        self.img_channels = 3
        self.depths = [args.zdim, 256, 256, 256, 128, 128]
        self.didx = 0
        self.alpha = 1.

        # init G
        self.G = nn.ModuleList()
        blk = nn.ModuleList()
        blk.append(ll.Conv2d(self.depths[0], self.depths[0], 4, padding=3)) # to 4x4
        blk.append(ll.Conv2d(self.depths[0], self.depths[0], 3, padding=1))
        self.G.append(blk)
        self.toRGB = nn.ModuleList()
        self.toRGB.append(ll.Conv2d(self.depths[0], self.img_channels, 1, lrelu=False, pnorm=False)) # toRGB

        # init D
        self.fromRGB = nn.ModuleList()
        self.fromRGB.append(ll.Conv2d(self.img_channels, self.depths[0], 1)) # fromRGB
        self.D = nn.ModuleList()
        blk = nn.ModuleList()
        blk.append(ll.MinibatchStddev())
        blk.append(ll.Conv2d(self.depths[0]+1, self.depths[0], 3, padding=1))
        blk.append(ll.Conv2d(self.depths[0], self.depths[0], 4, stride=4)) # to 1x1
        blk.append(ll.Flatten())
        blk.append(ll.Linear(self.depths[0], 1))
        self.D.append(blk)

        self.doubling = nn.Upsample(scale_factor=2)
        self.halving = nn.AvgPool2d(2, 2)
        self.set_optimizer() # 
        self.criterion = losses.GANLoss(loss_type=args.loss_type, device=device)
        self.loss_type = args.loss_type
    
    def generate(self, z):
        hz = z
        for idx in range(len(self.G)):
            for net in self.G[idx]:
                hz = net(hz)
            if idx == len(self.G)-2:
                res = hz
        xf = self.toRGB[self.didx](hz)
        if self.alpha < 1.0:
            res = self.toRGB[self.didx-1](res)
            res = self.doubling(res)
            xf = (1-self.alpha)*res + self.alpha*xf
        return xf
    
    def discriminate(self, x):
        nD = len(self.D)
        hy = self.fromRGB[self.didx](x)
        if self.alpha < 1.0:
            res = self.halving(x)
            res = self.fromRGB[self.didx-1](res)
        for idx in range(nD):
            for net in self.D[-idx-1]:
                hy = net(hy)
            if idx == 0 and self.alpha < 1.0:
                hy = (1-self.alpha)*res + self.alpha*hy
        y = hy
        return y
    
    def train_step(self, z, x):
        ## Training D
        self.D_opt.zero_grad()
        xf = self.generate(z)
        
        yr = self.discriminate(x)
        
        yf = self.discriminate(xf.detach())

        dloss_r = self.criterion(yr, True)
        dloss_f = self.criterion(yf, False)
        dloss = dloss_r + dloss_f
        if self.loss_type == 'wgan-gp':
            gp = 10.*self._gp(x, xf.detach())
            dloss = dloss + gp
        dloss.backward(retain_graph=True)
        self.D_opt.step()

        ## Training G
        self.G_opt.zero_grad()
        yf = self.discriminate(xf)

        gloss = self.criterion(yf, True)
        gloss.backward()
        self.G_opt.step()

        # for log
        training_info = {
            'Dloss': dloss.item(),
            'Dloss_r': dloss_r.item(),
            'Dloss_f': dloss_f.item(),
            'Gloss': gloss.item(),
            'gp': gp.item(),
        }
        return training_info

    def add_scale(self, increase_idx=True):
        if increase_idx:
            self.didx += 1
        blk = nn.ModuleList()
        blk.append(nn.Upsample(scale_factor=2))
        blk.append(
            ll.Conv2d(self.depths[self.didx-1], self.depths[self.didx], 3, padding=1)
        )
        blk.append(
            ll.Conv2d(self.depths[self.didx], self.depths[self.didx], 3, padding=1)
        )
        self.G.append(blk)
        self.toRGB.append(ll.Conv2d(self.depths[self.didx], self.img_channels, 1, lrelu=False, pnorm=False)) # toRGB

        self.fromRGB.append(ll.Conv2d(self.img_channels, self.depths[self.didx], 1)) # fromRGB
        blk = nn.ModuleList()
        blk.append(
            ll.Conv2d(self.depths[self.didx], self.depths[self.didx], 3, padding=1)
        )
        blk.append(
            ll.Conv2d(self.depths[self.didx], self.depths[self.didx-1], 3, padding=1)
        )
        blk.append(
            nn.AvgPool2d(2, stride=2)
        )
        self.D.append(blk)
        self.to(self.device)
        self.set_optimizer()
        self.set_alpha(0.)
        
    def set_optimizer(self):
        dparams = list(self.D.parameters()) + list(self.fromRGB.parameters())
        gparams = list(self.G.parameters()) + list(self.toRGB.parameters())
        self.D_opt = optim.Adam(
            filter(lambda p: p.requires_grad, dparams),betas=[0., 0.99], lr=self.args.lr)
        self.G_opt = optim.Adam(
            filter(lambda p: p.requires_grad, gparams),betas=[0., 0.99], lr=self.args.lr)

    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def _gp(self, x, xf):
        N, C, H, W = x.size()
        eps = torch.rand(N, 1, 1, 1)
        eps = eps.expand(-1, C, H, W).to(self.device)
        interpolates = eps*x + (1-eps)*xf
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        yi = self.discriminate(interpolates)
        yi = yi.sum()

        yi_grad = torch.autograd.grad(outputs=yi, inputs=interpolates,
                                        create_graph=True, retain_graph=True)

        yi_grad = yi_grad[0].view(N, -1)
        yi_grad = torch.norm(yi_grad, p=2, dim=1)
        gp = torch.pow(yi_grad-1., 2).sum()
        return gp

