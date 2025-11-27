import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage
import os

# import pywt

from matplotlib.colors import LinearSegmentedColormap

from scipy.ndimage import convolve
from torch.fft import fftn, ifftn, fftshift, ifftshift, rfftn, irfftn, rfft
from typing import Union


dtype = torch.cuda.FloatTensor


def custom_div_cmap(
    numcolors=11, name="custom_div_cmap", mincol="black", midcol="k", maxcol="green"
):
    # from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        name=name, colors=[mincol, midcol, maxcol], N=numcolors
    )
    return cmap


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
        return x[:, pad_len[1] // 2 : -pad_len[1] // 2, pad_len[2] // 2 : -pad_len[2] // 2]
    elif pad_len[0] != 0 and pad_len[1] == 0 and pad_len[2] == 0:
        return x[pad_len[0] // 2 : -pad_len[0] // 2, :, :]
    elif all([i == 0 for i in pad_len]):
        return x
