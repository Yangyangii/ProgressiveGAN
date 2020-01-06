import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self, loss_type='lsgan', device='cuda'):
        super(GANLoss, self).__init__()
        self.loss_type = loss_type
        real_label = torch.tensor(1.0).to(device)
        fake_label = torch.tensor(0.0).to(device)
        if loss_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.loss = nn.BCELoss()
        elif loss_type == 'wgan-gp':
            self.loss = wasserstein_loss
            fake_label = torch.tensor(-1.0).to(device)
        self.register_buffer('real_label', real_label)
        self.register_buffer('fake_label', fake_label)
    
    def __call__(self, inputs, is_real=None):
        if is_real is not None:
            labels = self.real_label.expand_as(inputs) if is_real else self.fake_label.expand_as(inputs)
            loss = self.loss(inputs, labels)
        #### Removed some codes ####
        return loss

def wasserstein_loss(inputs, labels):
    return torch.sum(labels*-inputs)