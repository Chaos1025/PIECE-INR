"""
File to implement image quality metrics for 3D images.
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
        mse = torch.mean((gt[i, ...] - img[i, ...]) ** 2)
        psnr_list[i] = -10 * torch.log10(mse)
    return psnr_list


class lpips_metric:
    def __init__(self, device=DEVICE):
        self.lpips_vgg = lpips.LPIPS(net="vgg").to(device)
        self.lpips_alex = lpips.LPIPS(net="alex").to(device)

    def compute(self, img1: torch.Tensor, img2: torch.Tensor):
        try:
            d = img1.shape[0]
            lpips_list = np.zeros((d, 2))
            for i in range(d):
                vgg = self.lpips_vgg(img1[i, ...], img2[i, ...]).mean().item()
                alex = self.lpips_alex(img1[i, ...], img2[i, ...]).mean().item()
                lpips_list[i] = np.array([vgg, alex])
            return np.array(lpips_list)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory")
            else:
                raise e

    def get_metric(self, gt: torch.Tensor, img: torch.Tensor, img_name: str = "", layer_range=None):
        assert (
            gt.shape == img.shape
        ), "Two images must have the same shape, but got {} and {}".format(gt.shape, img.shape)
        lpips_d = self.compute(gt, img)
        vgg_list = lpips_d[:, 0]
        alex_list = lpips_d[:, 1]
        if layer_range is not None:
            assert (
                layer_range[0] >= 0 and layer_range[1] <= gt.shape[0]
            ), "layer_range must be in [0, {}], but got {}".format(gt.shape[0], layer_range)
            vgg_list = vgg_list[layer_range[0] : layer_range[1]]
            alex_list = alex_list[layer_range[0] : layer_range[1]]

        print(
            f"{img_name} LPIPS vs GT: VGG={vgg_list.mean():.4e}, Alex={alex_list.mean():.4e}, the smaller the better"
        )
        return [vgg_list.mean(), alex_list.mean()]

    def get_slice_metric(self, gt: torch.Tensor, img: torch.Tensor, img_name: str = ""):
        assert (
            gt.shape == img.shape
        ), "Two images must have the same shape, but got {} and {}".format(gt.shape, img.shape)
        lpips_d = self.compute(gt, img)
        return lpips_d


class reconstruction_metric:
    def __init__(self, device=DEVICE):
        self.lpips = lpips_metric(device=device)
        self.mse = nn.MSELoss()
        self.layer_wise_PSNR = layer_wise_PSNR
        self.ssim = ssim_d
        self.device = device

    def get_metric(self, gt: np.array, img: np.array, img_name: str = ""):
        assert (
            gt.shape == img.shape
        ), "Two images must have the same shape, but got {} and {}".format(gt.shape, img.shape)
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
        print(f"{img_name} PSNR vs GT: {-10*np.log10(mse_d):.4f}dB, the larger the better")
        print(f"{img_name} PSNR layer-wise vs GT: {PSNR_layer_mean:.4f}dB, the larger the better")
        print(f"{img_name} SSIM vs GT: {ssim_d:.4f}, the larger the better")
        lpips_d = self.lpips.get_metric(gt, img, img_name)
        return mse_d, ssim_d, lpips_d
