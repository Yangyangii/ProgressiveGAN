import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope, inplace=inplace)

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * torch.rsqrt((x**2).mean(dim=1, keepdim=True) + epsilon)

class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()

    def forward(self, x):
        N, _, H, W = x.size()
        std = torch.std(x, dim=0, keepdim=True) # (1, C, H, W)
        std_mean = torch.mean(std, dim=(1,2,3), keepdim=True).expand(N, -1, H, W)
        # std_mean = torch.mean(std, dim=1, keepdim=True).expand(x.size(0), -1, -1, -1)
        return torch.cat([x, std_mean], dim=1) # (N, C+1, H, W)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1,
                    lrelu=True, weight_norm=False, pnorm=True, equalized=True):
        super(Conv2d, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2 * dilation
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups
        )
        if weight_norm:
            nn.utils.weight_norm(self.conv)
        
        self.lrelu = LeakyReLU() if lrelu else None
        self.normalize = PixelNorm() if pnorm else None
        self.equalized = equalized
        if equalized:
            self.conv.weight.data.normal_(0, 1)
            fan_in = np.prod(self.conv.weight.size()[1:])
            self.he_constant = np.sqrt(2.0/fan_in)
            self.conv.bias.data.fill_(0.)
        
    def forward(self, x): 
        y = self.conv(x)
        y = y*self.he_constant if self.equalized else y
        y = self.lrelu(y) if self.lrelu is not None else y
        y = self.normalize(y) if self.normalize is not None else y
        return y

class Linear(nn.Module):
    def __init__(self, in_dims, out_dims, 
                weight_norm=False, equalized=True):
        super(Linear, self).__init__()
        
        self.linear = nn.Linear(in_dims, out_dims)
        if weight_norm:
            nn.utils.weight_norm(self.linear)
        self.equalized = equalized
        if equalized:
            self.linear.weight.data.normal_(0, 1)
            fan_in = np.prod(self.linear.weight.size()[1:])
            self.he_constant = np.sqrt(2.0/fan_in)
            self.linear.bias.data.fill_(0.)

    def forward(self, x): 
        y = self.linear(x)
        y = y*self.he_constant if self.equalized else y
        return y
