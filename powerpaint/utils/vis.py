import os
import torch
import torchvision

def save_img(path, img_01):  # [3,H,W] in [0,1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(img_01.clamp(0,1), path)

def save_gray(path, x):  # [1,H,W]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x.detach()
    vmin = torch.quantile(x.flatten(), 0.01)
    vmax = torch.quantile(x.flatten(), 0.99)
    x = (x - vmin) / (vmax - vmin + 1e-8)
    torchvision.utils.save_image(x.clamp(0,1), path)

def save_grid(path, imgs_01, nrow=4):  # [N,3,H,W]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = torchvision.utils.make_grid(imgs_01.clamp(0,1), nrow=nrow)
    torchvision.utils.save_image(grid, path)