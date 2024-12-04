""" File to implement SSIM and mSSIM metrics for 3D images.

"""

import torch
import numpy as np
import lpips
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ssim_d(img1, img2):
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_mu2 = mu1 * mu2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = torch.mean(img1 * img1) - mu1_sq
    sigma2_sq = torch.mean(img2 * img2) - mu2_sq
    sigma12 = torch.mean(img1 * img2) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / torch.clamp(
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2), min=1e-8, max=1e8
        )
    

    return loss


def layer_wise_PSNR(gt: torch.Tensor, img: torch.Tensor):
    assert gt.shape == img.shape, "Two images must have the same shape"
    d = gt.shape[0]
    psnr_list = torch.zeros(d)
    for i in range(d):
        mse = torch.mean((gt[i, ...] - img[i, ...])**2)
        psnr_list[i] = -10*torch.log10(mse)
    return psnr_list

class lpips_metric:
    def __init__(self, device=DEVICE):
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device)
        self.lpips_alex = lpips.LPIPS(net='alex').to(device)

    def compute(self, img1: torch.Tensor, img2: torch.Tensor):         
        try:
            vgg = self.lpips_vgg(img1, img2).mean().item()
            alex = self.lpips_alex(img1, img2).mean().item()
            return np.array([vgg, alex])
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory")
                d = img1.shape[0] 
                lpips_list = np.zeros((d, 2))
                for i in range(d):
                    lpips_list[i] = self.compute(img1[i, ...], img2[i, ...])
                return lpips_list.mean(axis=0)
            else:
                raise e

    def get_metric(self, gt: torch.Tensor, img: torch.Tensor, img_name: str =""):
        lpips_d = self.compute(gt, img)
        print(f"{img_name} LPIPS vs GT: VGG={lpips_d[0]:.4e}, Alex={lpips_d[1]:.4e}, the smaller the better")
        return lpips_d

class reconstruction_metric:
    def __init__(self, device=DEVICE):
        self.lpips = lpips_metric(device=device)
        self.mse = nn.MSELoss()
        self.layer_wise_PSNR = layer_wise_PSNR
        self.ssim = ssim_d
        self.device = device

    def get_metric(self, gt: np.array, img: np.array, img_name: str =""):
        assert gt.shape == img.shape, "Two images must have the same shape, but got {} and {}".format(gt.shape, img.shape)
        gt = torch.tensor(gt).float().to(self.device)
        img = torch.tensor(img).float().to(self.device)
        if len(gt.shape) == 3:
            gt = gt.unsqueeze(1)
            img = img.unsqueeze(1)
        if len(gt.shape) == 2:
            gt = gt.unsqueeze(0).unsqueeze(0)
            img = img.unsqueeze(0).unsqueeze(0)

        PSNR_layer_wise = self.layer_wise_PSNR(gt, img)
        PSNR_layer_mean = PSNR_layer_wise.mean().item()
        mse_d = self.mse(gt, img).item()
        ssim_d = self.ssim(gt, img).item()
        lpips_d = self.lpips.get_metric(gt, img, img_name)
        print(f"{img_name} PSNR vs GT: {-10*np.log10(mse_d):.4f}dB, the larger the better")
        print(f"{img_name} PSNR layer-wise vs GT: {PSNR_layer_mean:.4f}dB, the larger the better")
        print(f"{img_name} SSIM vs GT: {ssim_d:.4f}, the larger the better")
        return mse_d, ssim_d, lpips_d

