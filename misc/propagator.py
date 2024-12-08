""" 
File for physical propagation.
"""

import torch
import torch.nn.functional as F
import functools
from torch.fft import fftn, ifftn, fftshift, ifftshift

get_half_even = lambda x: (x) // 2 if x % 2 == 0 else (x + 1) // 2


class PhysicalPropagator:
    def __init__(self, psf: torch.Tensor, obj_shape: tuple, psf_shape: tuple):
        self.obj_shape = obj_shape
        self.psf_shape = psf_shape
        print("obj_shape in PP : ", obj_shape)
        print("psf_shape in PP : ", psf_shape)
        assert all(
            [psf_shape[i] >= psf.shape[i] for i in range(3)]
        ), "Provided PSF shape should be larger than the PSF's"

        d, h, w = (get_half_even(i) for i in self.obj_shape)
        self.psf_padder = functools.partial(F.pad, pad=(w, w, h, h, d, d), mode="constant", value=0)
        
        d, h, w = (get_half_even(i) for i in self.psf_shape)
        self.obj_padder = functools.partial(F.pad, pad=(w, w, h, h, d, d), mode="constant", value=0)
        
        self.otf = self.psf2otf(psf)


    def psf2otf(self, psf: torch.Tensor) -> torch.Tensor:
        """Change PSF in spatial domain to OTF in frequency domain
        Return: OTF, shape of `psf`+`obj`
        """
        psf_pad = self.psf_padder(psf)
        otf = fftshift(fftn(ifftshift(psf_pad)))
        return otf

    def propagate(self, obj: torch.Tensor, y_max=1):
        """Propagate object `obj` with the OTF"""
        d, h, w = (get_half_even(i) for i in self.psf_shape)  # half of the shape

        obj = obj.view(1, *obj.shape)
        obj_pad = self.obj_padder(obj)
        obj_pad = obj_pad.squeeze(0)
        obj_ft = fftshift(fftn(ifftshift(obj_pad)))
        # breakpoint()
        y_pad = torch.real(fftshift(ifftn(ifftshift(obj_ft * self.otf))))
        y_ = y_pad[d:-d, h:-h, w:-w]
        
        return y_ / y_max
