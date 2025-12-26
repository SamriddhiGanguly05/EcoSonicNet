# utils.py — Basic helpers for HTSAT

import torch
import torch.nn.functional as F

def do_mixup(x, y, alpha=0.4):
    '''Simple Mixup augmentation'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def interpolate(x, ratio):
    '''Resize 2D tensor along last dimension'''
    return F.interpolate(x.unsqueeze(1), scale_factor=ratio, mode='nearest').squeeze(1)
