import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from tensorboardX import SummaryWriter

import numpy as np
import datetime
import os, sys
import glob
from tqdm import tqdm
from PIL import Image
from matplotlib.pyplot import imshow, imsave
from torchvision.utils import save_image

from pggan import PGGAN

from config import ConfigArgs as args

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIM = (128, 128, 3)

def tensor2img(tensor):
    img = (np.transpose(tensor.detach().cpu().numpy(), [1,2,0])+1)/2.
    img = np.clip(img, 0, 1)
    return img

def train():
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(args.log_dir, 'board'))

    transform = transforms.Compose([
        transforms.Resize((IMAGE_DIM[0],IMAGE_DIM[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(
        root=args.data_dir, transform=transform
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=8, pin_memory=True
    )

    gan = PGGAN(args).to(DEVICE)
    global_step = 0
    if args.retrain:
        model = torch.load(args.load_model)
        global_step = model['global_step']
        global_step -= 1
        gan.didx = model['didx']
        for idx in range(gan.didx):
            gan.add_scale(increase_idx=False)
        gan.set_alpha(model['alpha'])
        print('global step: {}, Resolution: {}'.format(global_step, 4*2**gan.didx))
        gan.G.load_state_dict(model['G'])
        gan.D.load_state_dict(model['D'])
        gan.G_opt.load_state_dict(model['G_opt'])
        gan.D_opt.load_state_dict(model['D_opt'])
        gan.toRGB.load_state_dict(model['toRGB'])
        gan.fromRGB.load_state_dict(model['fromRGB'])
        
    nrows = 8 # for samples
    nrows_r = int(np.sqrt(args.batch_size))
    fix_z = torch.randn([nrows**2, args.zdim, 1, 1]).to(DEVICE)
    epochs = 0
    next_scale_step = 0
    alpha = 1.
    for i in range(gan.didx+1):
        next_scale_step += sum(args.scale_update_schedule[i])
    # next_scale_step = sum(args.scale_update_schedule[gan.didx])
    next_alpha_step = 0

    while global_step < args.max_step:
        with tqdm(enumerate(data_loader), total=len(data_loader), ncols=70) as t:
            t.set_description('{}x{}, Step {}, a {}'.format(4*2**gan.didx,4*2**gan.didx,global_step, alpha))
            for idx, (images, _) in t:
                stride = 2**(5-gan.didx)
                x = images[..., ::stride, ::stride].to(DEVICE)
                z = torch.randn([x.size(0), args.zdim, 1, 1]).to(DEVICE)
                tinfo = gan.train_step(z, x)

                global_step += 1

                if global_step % args.log_step == 0:
                    for k, v in tinfo.items():
                        writer.add_scalar(k, v, global_step=global_step)

                    gan.eval()
                    with torch.no_grad():
                        xf = gan.generate(fix_z)
                    xf = torch.cat([torch.cat([xf[nrows*j+i] for i in range(nrows)], dim=1) for j in range(nrows)], dim=2)
                    imgs = tensor2img(xf)
                    imsave('{}/{:04d}k.jpg'.format(args.sample_dir, global_step//1000), imgs)
                    xr = torch.cat([torch.cat([x[nrows_r*j+i] for i in range(nrows_r)], dim=1) for j in range(nrows_r)], dim=2)
                    imgs = tensor2img(xr)
                    imsave('{}/{:04d}k-real.jpg'.format(args.sample_dir, global_step//1000), imgs)
                    # writer.add_image()
                    gan.train()

                if global_step % args.save_step == 0:
                    save_model(gan, global_step)

                if global_step == next_scale_step and stride > 1:
                    print('\nScale up\n')
                    gan.add_scale()
                    next_scale_step = global_step + sum(args.scale_update_schedule[gan.didx])
                    alpha_idx = 0
                    alpha = args.scale_update_alpha[gan.didx][alpha_idx]
                    gan.set_alpha(alpha)
                    next_alpha_step = global_step + args.scale_update_schedule[gan.didx][alpha_idx]
                elif global_step == next_alpha_step:
                    alpha_idx += 1
                    alpha = args.scale_update_alpha[gan.didx][alpha_idx]
                    gan.set_alpha(alpha)
                    next_alpha_step = global_step + args.scale_update_schedule[gan.didx][alpha_idx]
                

            epochs += 1
        
            
def save_model(model, global_step):
    infos = {
        'G': model.G.state_dict(),
        'D': model.D.state_dict(),
        'G_opt': model.G_opt.state_dict(),
        'D_opt': model.D_opt.state_dict(),
        'toRGB': model.toRGB.state_dict(),
        'fromRGB': model.fromRGB.state_dict(),
        'global_step': global_step,
        'didx': model.didx,
        'alpha': model.alpha,
    }
    torch.save(infos, '{}/{}-{:04d}k.pth.tar'.format(args.ckpt_dir, type(model).__name__, global_step//1000))


if __name__ == "__main__":
    train()