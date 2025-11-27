"""
Manage physical parameters of optical system
"""

import numpy as np


# Physical parameters
class PhysicsParamsManager(object):
    def __init__(self, args):
        super(PhysicsParamsManager, self).__init__()
        self.dz = args.psf_dz
        self.dy = args.psf_dy
        self.dx = args.psf_dx
        self.NA = args.NA  # Numerical Aperture
        self.ni = args.ni  # Refractive index of the immersion medium
        self.ns = args.ns  # Refractive index of the object
        self.wave_length = args.wl  # Emission light wave length (um)
        self.alpha = np.arcsin(
            self.NA / self.ni
        )  # Semi-angle of the cone of light that enters the objective lens
        self.k_prime = self.ni / self.wave_length
        self.r_cut_freq = 2 * self.NA / self.wave_length
        self.z_cut_freq = self.ni * (1 - np.cos(self.alpha)) / self.wave_length

        return

    def PSF_params(self, PSF_shape: tuple = None) -> dict:
        params = {
            "units": (self.dz, self.dy, self.dx),
            "NA": self.NA,
            "ni": self.ni,
            "ns": self.ns,
            "wave_length": self.wave_length,
        }
        return params

    def WIN_params(self, sample_shape: tuple) -> dict:
        z_range, y_range, x_range = sample_shape
        dfz = 1 / (z_range * self.dz)
        dfy = 1 / (y_range * self.dy)
        dfx = 1 / (x_range * self.dx)
        params = {
            "units": (self.dz, self.dy, self.dx),
            "NA": self.NA,
            "ni": self.ni,
            "ns": self.ns,
            "wave_length": self.wave_length,
            "freq_units": (dfz, dfy, dfx),
            "z_cut_freq": self.z_cut_freq,
            "y_cut_freq": self.r_cut_freq,
            "x_cut_freq": self.r_cut_freq,
        }
        return params

    def ENC_params(self, coord_shape: tuple) -> dict:
        d, h, w = coord_shape
        dfz = 1 / (d * self.dz)
        dfy = 1 / (h * self.dy)
        dfx = 1 / (w * self.dx)
        params = {
            "fz_cut": self.z_cut_freq / dfz,
            "fy_cut": self.r_cut_freq / dfy,
            "fx_cut": self.r_cut_freq / dfx,
        }
        return params
