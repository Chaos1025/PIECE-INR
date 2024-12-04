import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from torch.fft import fftn, ifftn, fftshift, ifftshift, rfftn, irfftn, rfft
from scipy.ndimage import convolve
from typing import Union

#* There isn't a defination of variable 'dtype' in raw code, I copy the def from main script.
dtype = torch.cuda.FloatTensor 


def crop_stack(x, pad_len: Union[tuple, list]):
    # crop the sub-area in input stack into the size of measurement
    # return the center area of input stack, with the same size as measurement
    if all([i != 0 for i in pad_len]):
        return x[
            pad_len[0] // 2 : -pad_len[0] // 2,
            pad_len[1] // 2 : -pad_len[1] // 2,
            pad_len[2] // 2 : -pad_len[2] // 2,
        ]
    elif pad_len[0] == 0 and pad_len[1] != 0 and pad_len[2] != 0:
        return x[
            :, pad_len[1] // 2 : -pad_len[1] // 2, pad_len[2] // 2 : -pad_len[2] // 2
        ]
    elif pad_len[0] != 0 and pad_len[1] == 0 and pad_len[2] == 0:
        return x[pad_len[0] // 2 : -pad_len[0] // 2, :, :]
    elif all([i == 0 for i in pad_len]):
        return x
    

def fill_stack(x, pad_len: Union[tuple, list]):
    # fill the input stack with zeros, with the size of measurement
    # return the filled stack
    return F.pad(
        x,
        (pad_len[0] // 2, pad_len[0] // 2, pad_len[1] // 2, pad_len[1] // 2, pad_len[2] // 2, pad_len[2] // 2),
        mode="constant",
        value=0,
    )

def get_measured_area(x: torch.Tensor, pad_len: Union[tuple, list]) -> torch.Tensor:
    # return the sub-area of measurement in the input stack, with the same size as the input stack
    # only Ture value in the area of measurement, the rest are False
    whole_area = torch.zeros_like(x, dtype=torch.bool)
    if all([i != 0 for i in pad_len]):
        whole_area[
            pad_len[0] // 2 : -pad_len[0] // 2,
            pad_len[1] // 2 : -pad_len[1] // 2,
            pad_len[2] // 2 : -pad_len[2] // 2,
        ] = True
    elif pad_len[0] == 0 and pad_len[1] != 0 and pad_len[2] != 0:
        whole_area[
            :, pad_len[1] // 2 : -pad_len[1] // 2, pad_len[2] // 2 : -pad_len[2] // 2
        ] = True
    elif pad_len[0] != 0 and pad_len[1] == 0 and pad_len[2] == 0:
        whole_area[pad_len[0] // 2 : -pad_len[0] // 2, :, :] = True
    elif all([i == 0 for i in pad_len]):
        whole_area = torch.ones_like(x, dtype=torch.bool)
    return whole_area


def save_args(args, exp_dir):
    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


def fft_convolve(im1, im2, mode='fftn'):
    if mode == 'fftn':
        _im1 = fftshift(fftn(ifftshift(im1)))
        _im2 = fftshift(fftn(ifftshift(im2)))
        # _im = apply_window(_im1) * apply_window(_im2)
        _im = _im1 * _im2
        # _im = torch.einsum('ijk,ijk->ijk', _im1, _im2)
        
        return torch.real(fftshift(ifftn(ifftshift(_im))))
    
    elif mode == 'rfftn':
        _im1 = fftshift(rfftn(ifftshift(im1)))
        _im2 = fftshift(rfftn(ifftshift(im2)))
        _im = _im1 * _im2
        # _im = apply_window(_im1) * apply_window(_im2)
        # _im = _im1 * _im2

        return fftshift(irfftn(ifftshift(_im)))


# Calculate the metric of an image in focus, 
# 'Tenengrad', 'Laplacian', 'Variance', 'Vollath'
def calculate_infocus_metrics(image, method='vollath'):
    if method == 'tenengrad':
        # Tenengrad focus measure
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gx = convolve(image.astype('float'), sobel_x)
        gy = convolve(image.astype('float'), sobel_y)
        tenengrad = np.mean(gx**2 + gy**2)
        return tenengrad

    elif method == 'laplacian':
        # Laplacian focus measure
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        laplacian = convolve(image.astype('float'), laplacian_kernel)
        laplacian_focus_measure = np.var(laplacian)
        return laplacian_focus_measure

    elif method == 'valv':
        # Variance of absolute values of Laplacian - VALV (another form of laplacian focus measure)
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        laplacian = convolve(image.astype('float'), laplacian_kernel)
        valv = np.var(np.abs(laplacian))
        return valv
    
    elif method == 'vollath':
        # Vollath's F4 focus measure
        image_double = image.astype('float')
        vollath = np.mean(image_double[:-1,:] * image_double[1:,:]) - np.mean(image_double[:-1,:])
        return vollath

    else:
        raise ValueError("Unknown method: {}".format(method))

