from curses import window
import torch
import torch.signal.windows as win
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dtype = torch.cuda.FloatTensor


def hessian_loss(img: torch.Tensor, sigma):
    """Hessian matrix calculation
    img: d x h x w
    sigma: scale along z(d) axis
    """
    sigma = torch.Tensor([sigma]).type(torch.float32).to(img.device)
    fun = torch.abs
    dz = img[1:, :, :] - img[:-1, :, :]  # [d-1, h, w]
    dx = img[:, 1:, :] - img[:, :-1, :]  # [d, h-1, w]
    dy = img[:, :, 1:] - img[:, :, :-1]  # [d, h, w-1]

    dzz = torch.mean(fun(dz[1:, :, :] - dz[:-1, :, :]))  # [d-2, h, w]
    dxx = torch.mean(fun(dx[:, 1:, :] - dx[:, :-1, :]))  # [d, h-2, w]
    dyy = torch.mean(fun(dy[:, :, 1:] - dy[:, :, :-1]))  # [d, h, w-2]

    dxy = torch.mean(fun(dx[:, :, 1:] - dx[:, :, :-1]))  # [d, h-1, w-1]
    dxz = torch.mean(fun(dx[1:, :, :] - dx[:-1, :, :]))  # [d-1, h, w-1]
    dyz = torch.mean(fun(dy[1:, :, :] - dy[:-1, :, :]))  # [d-1, h-1, w]

    hessian = (
        dxx
        + dyy
        + sigma * dzz
        + 2 * dxy
        + 2 * torch.sqrt(sigma) * dxz
        + 2 * torch.sqrt(sigma) * dyz
    )
    return hessian


def mse_loss(y_pred, y_true):
    return torch.mean(torch.square(y_pred - y_true))


def mae_loss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def ssim_loss(img1, img2):
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

    loss = 1.0 - (
        ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        / torch.clamp((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2), min=1e-8, max=1e8)
    )

    return loss.type(dtype)


class GFDMAE_Loss(torch.nn.Module):
    """Calculate the Fourier Domain Mean Absolute Error
    between pred and targets
    """

    def __init__(self, shape: tuple, phys_params: dict, device="cpu"):
        super(GFDMAE_Loss, self).__init__()
        window_func = win.gaussian
        funits = phys_params["freq_units"]
        z_cut_freq = phys_params["z_cut_freq"]
        y_cut_freq = phys_params["y_cut_freq"]
        x_cut_freq = phys_params["x_cut_freq"]
        # const = np.sqrt(2 * np.log(2))
        const = 3
        std_z = int(z_cut_freq / funits[0]) / const
        std_y = int(y_cut_freq / funits[1]) / const
        std_x = int(x_cut_freq / funits[2]) / const

        if len(shape) == 2:
            h, w = shape
            self.window = window_func(h, std=std_y).unsqueeze(1) * window_func(
                w, std=std_x
            ).unsqueeze(0)
        elif len(shape) == 3:
            d, h, w = shape
            self.window = (
                window_func(d, std=std_z).unsqueeze(1).unsqueeze(1)
                * window_func(h, std=std_y).unsqueeze(1)
                * window_func(w, std=std_x).unsqueeze(0)
            )
        else:
            raise ValueError("Invalid shape for GFDMAE_Loss")
        self.window = self.window.to(device)

    def forward(self, img1, img2):
        F1 = torch.fft.fftshift(torch.fft.fftn(img1))
        F2 = torch.fft.fftshift(torch.fft.fftn(img2))
        return mae_loss(F1 * self.window, F2 * self.window)
