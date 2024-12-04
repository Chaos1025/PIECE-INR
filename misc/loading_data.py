# Author: Chenyu Xu
# Last Update: 2023/11/07
# Version: 1.1
# Description:
#   File to load data for the `bf_cocoa_rec3d.py` scripts.
#   2 tpye of imaging modes, 'simulator' or 'solver':
#       'simulator' mode simulates a measurement from a known 3d psf and 3d ref-object,
#           and return measurement, psf, ref-object, and the maximum value of the measurement.
#       'solver' mode load a measurement from a real dataset, and return measurement.
#   2 type of psf generation, 'Gibson' or 'Angular':
#       'Gibson' mode generate a 3d psf using Gibson-Lanni model.
#       'Angular' mode generate a 3d psf using angular spectrum method.
#   For imaging model, in 'simulator' mode, you can invert ref or not,
#       where we can simulate a bright field measurement or just fluorescence measurement.
# * Note that the background value is 0 in fluorescence microscope,
# * but 1 or other `bg_value` we choose for bright field microscope.
#       In 'solver' mode, you can invert sample or not,
#       which depends on whether solve deconvolution from measurement or its inverse.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import tifffile as tiff

from .psf_torch import PsfGenerator3D
import microscPSF.microscPSF as msPSF
from .utils import fft_convolve
from scipy.io import loadmat
from typing import Union

dtype = torch.cuda.FloatTensor


def prop_obj(
    obj: Union[torch.Tensor, np.ndarray],
    psf: Union[torch.Tensor, np.ndarray],
    pad_value: float = 0,
):
    """Generate measurements from a known 3d psf and 3d ref-object.
    Spread the object with psf, a forward physical process.
    Calculate the convolution of object and psf.
    achieve it by fft method, on cpu.
    """
    if type(obj) is np.ndarray:
        obj = torch.from_numpy(obj).type(dtype)
    if type(psf) is np.ndarray:
        psf = torch.from_numpy(psf).type(dtype)

    d, w, h = psf.shape  # [z, x, y]
    obj_pad = F.pad(
        obj, (h, h, w, w, d, d), mode="constant", value=pad_value
    )  # pad along the 1st dim, z-axis
    psf_pad = F.pad(psf, (h, h, w, w, d, d), mode="constant", value=0)
    y_pad = fft_convolve(obj_pad, psf_pad, mode="fftn")
    y_ = y_pad[d:-d, w:-w, h:-h]
    y_max = y_.max().item()
    y_ = y_ / y_max
    return y_, y_max


def check_shape(kernel_shape: list, sample_shape: list):
    """Check for proper shape of kernel and sample
    Kernel shape plus sample shape should not larger than three times of sample shape,
    and the difference between kernel shape and sample shape should be even.
    """
    for i in range(len(kernel_shape)):
        assert (
            kernel_shape[i] + sample_shape[i] + 1 <= 3 * sample_shape[i]
        ), "The sum of ker and samp is larger than 3 times of samp in dim-{}".format(i)
        assert (
            kernel_shape[i] - sample_shape[i]
        ) % 2 == 0, "The diff of ker and samp in dim-{} is not even".format(i)
    return None


def adjust_matrix_shape(mat: np.ndarray, shape: list) -> np.ndarray:
    for i in range(len(shape)):
        diff = mat.shape[i] - shape[i]
        if diff > 0:  # If mat's dimension is larger, trim it
            trim_size = diff // 2
            mat = np.take(mat, range(trim_size, trim_size + shape[i]), axis=i)
        elif diff < 0:  # If mat's dimension is smaller, pad it
            pad_size = abs(diff) // 2
            pad_width = [
                (0, 0) if j != i else (pad_size, pad_size + diff % 2)
                for j in range(len(shape))
            ]
            mat = np.pad(mat, pad_width, mode="constant")
    return mat


class MicroDataLoader:
    def __init__(
        self, mode, args, phys_params, measurement=None, init_guess=None, psf=None, ref=None, rl=None
    ):
        assert mode in ["simulator", "solver"]
        assert args.psf_generation in ["Gibson", "Angular", "External"]
        self.mode = mode
        self.dz = phys_params["units"][0]
        self.dy = phys_params["units"][1]
        self.dx = phys_params["units"][2]
        self.NA = phys_params["NA"]
        self.ns = phys_params["ns"]
        self.ni = phys_params["ni"]
        self.wave_length = phys_params["wave_length"]
        self.data_dir = os.path.join(args.root_dir, args.data_stack_name)
        self.ref_dir = os.path.join(args.root_dir, args.ref_name)
        self.psf_type = args.psf_generation
        self.psf_dir = args.psf_path
        self.normalized = args.normalized

        self.measurement = measurement
        self.init_data = init_guess
        self.psf = psf
        self.ref = ref
        self.rl_data = rl

    def load_data(self, data_dir):
        # loading corresponding .tif data
        data = tiff.imread(data_dir).astype(np.float32)
        # min_data = data.min()
        min_data = data.min()
        max_data = data.max()
        data = (data - min_data) / (
            max_data - min_data
        )  # Subtract minimum value (constant background) and normalize sample
        print("Normalize data from [{},{}] to [0,1]".format(min_data, max_data))
        return data

    def load_extern_psf(self, psf_shape, psf_dir: str) -> np.ndarray:
        # Loading PSF from external file, and adjust its shape if necessary
        if psf_dir.endswith(".tif"):
            psf_ = tiff.imread(psf_dir).astype(np.float32)
        elif psf_dir.endswith(".mat"):
            psf_ = loadmat(psf_dir)["psf"]
            # Matlab use a different sequence of axis
            psf_ = np.transpose(psf_, (2, 1, 0)).astype(np.float32)
        print("Load PSF from external file with shape of: ", psf_.shape)
        if psf_shape is not None and psf_.shape != psf_shape:
            assert all(
                [(psf_shape[i] - psf_.shape[i]) % 2 == 0 for i in range(3)]
            ), "The difference between loaded psf shape and regularized shape should be even in any axis."
            psf_ = adjust_matrix_shape(psf_, psf_shape)
        psf_ = psf_ / np.sum(psf_, axis=(1,2), keepdims=True) 
        psf_ = psf_ / np.sum(psf_)  # Normalizing the psf for conservation of energy
        return psf_

    def psf_space_constraint(self, psf_, r_ratio):
        """Set the psf to 0 outside a circle of radius r_ratio * psf_.shape[1]"""
        assert r_ratio <= 1
        d, w, h = psf_.shape
        r = int(r_ratio * (w // 2))
        center = (d // 2, w // 2, h // 2)
        kz = np.linspace(-center[0], d - center[0], d)
        kx = np.linspace(-center[1], w - center[1], w)
        ky = np.linspace(-center[2], h - center[2], h)
        z, x, y = np.meshgrid(kz, kx, ky, indexing="ij")
        mask = x * x + y * y <= r * r
        psf_[~mask] = 0
        return psf_

    def psf_freq_constraint(self, psf_, r_ratio):
        """Set the otf to 0 outside a circle of radius r_ratio * psf_.shape[1]"""
        #! Haven't be tested or used
        assert r_ratio <= 1
        d, w, h = psf_.shape
        r = int(r_ratio * (w // 2))
        center = (d // 2, w // 2, h // 2)
        kz = np.linspace(-center[0], d - center[0], d)
        kx = np.linspace(-center[1], w - center[1], w)
        ky = np.linspace(-center[2], h - center[2], h)
        z, x, y = np.meshgrid(kz, kx, ky, indexing="ij")
        mask = x * x + y * y <= r * r
        otf = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(psf_)))
        otf[~mask] = 0
        psf_ = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(otf)))
        return np.abs(psf_)

    def load(
        self,
        PSF_shape=None,
        invert_ref: bool = False,
        invert_sample: bool = False,
        psf_constraint: bool = False,
        r_ratio=1,
        bg_value=1,
    ) -> dict:
        """Load datas for recovery in bright field mode.
        In 'simulator' mode, the reference object is loaded from a tiff file,
            which is default a 3d numpy array with shape (d, w, h),
            captured using fluorescence microscope or confocal microscope.
        So if we want to simulate a bright field measurement, we need to
            whether invert the reference object, or invert the sample object.
        #! And it's a key point to pay attention to the background value,
            which is 0 in fluorescence microscope, 1 if we just invert ref.

        In 'solver' mode, the measurement is loaded from a tiff file,
            which is captured in real bright field microscope.

        Args:
            invert_ref: whether to invert the reference object.
            invert_sample: whether to invert the sample object.
            r_ratio: the ratio of the radius of the psf support.

        Returns:
            ret_pack: a dict containing the following keys:
                ref: the reference object, a 3d numpy array.
                psf: the psf, a 3d numpy array.
                y: the measurement, a 3d numpy array.
                y_max: the maximum value of the measurement.

        """
        # load data,
        ret_pack = {}

        print("Only work in solver mode!!")
        if self.measurement is None:
            y_ = self.load_data(self.data_dir)
        else:
            print("load measurement from input")
            y_ = self.measurement
        sample_shape = y_.shape
        y_max = y_.max()
        y_min = y_.min()

        # * Load PSF
        if self.psf_type == "Gibson":
            if PSF_shape is None:
                PSF_shape = sample_shape
            gibson_gen = gibsonPSFGenerator(
                PSF_shape,
                units=(self.dz, self.dy, self.dx),
                NA=self.NA,
                wave_length=self.wave_length,
                ns=self.ns,
                ni=self.ni,
            )
            psf_ = gibson_gen.PSF()
            psf_ = psf_ / np.sum(psf_)  # Normalizing the psf for conservation of energy
        elif self.psf_type == "Angular":
            if PSF_shape is None:
                PSF_shape = sample_shape
            psf_gen = PsfGenerator3D(
                psf_shape=PSF_shape,
                units=(self.dz, self.dy, self.dx),
                na_detection=self.NA,
                lam_detection=self.wave_length,
                n=self.ns,
            )
            psf_ = (
                psf_gen.incoherent_psf(
                    nn.parameter.Parameter(torch.tensor([0], dtype=torch.float32)),
                    self.normalized,
                )
                / sample_shape[0]
            )
            psf_ = psf_.detach().cpu().numpy()
        elif self.psf_type == "External":
            psf_ = self.load_extern_psf(
                PSF_shape, self.psf_dir
            )  # padding has been done inside the function
        else:
            raise NotImplementedError(
                f"PSF type {self.psf_type} has not been implemented."
            )

        if psf_constraint:
            psf_ = self.psf_space_constraint(psf_, r_ratio)
        if invert_sample:
            y_ = 1 - y_

        if os.path.exists(self.ref_dir) and self.ref is None:
            ret_pack["ref"] = self.load_data(self.ref_dir)
        elif self.ref is not None:
            ret_pack["ref"] = self.ref

        ret_pack["rl"] = self.rl_data

        ret_pack["psf"] = psf_ / psf_.sum()
        ret_pack["y"] = y_ / y_max
        ret_pack["y_max"] = y_max
        ret_pack["y_min"] = y_min * 0.95
        ret_pack["init_max"] = self.init_data.max()
        ret_pack["init_data"] = self.init_data / ret_pack["init_max"]
        ret_pack["y_shape"] = sample_shape

        print("PSF shape after loading: ", psf_.shape)
        print("Measurement shape after loading: ", y_.shape)
        print("Maximum value of measurement: ", y_max)
        print("Minimum value of measurement: ", y_min)

        return ret_pack


class gibsonPSFGenerator:
    def __init__(self, psf_shape, units, NA, wave_length, ns=1.333, ni=1):
        self.wave_length = wave_length
        self.units = units
        self.shape = psf_shape
        self.mp = msPSF.m_params
        self.mp["NA"] = NA
        self.mp["ns"] = ns
        self.mp["ni"] = ni
        self.mp["ni0"] = ni
        self.rv = np.linspace(
            0, (psf_shape[1] - 1) * units[1], psf_shape[1]
        )  # coordinates along lateral direction
        z_halfdepth = (psf_shape[0] - 1) * units[0] / 2
        self.zv = np.linspace(
            -z_halfdepth, z_halfdepth, psf_shape[0]
        )  # coordinates along axial direction

    def PSF(self):
        """Generate the 3D PSF"""
        psf = msPSF.gLXYZFocalScan(
            self.mp, self.units[1], self.shape[1], self.zv, wvl=self.wave_length
        )
        psf = psf / np.sum(psf, axis=(1, 2), keepdims=True)
        print(
            "Generate PSF with shape: ",
            psf.shape,
            " and units: ",
            self.units,
            ", using Gibson-Lanni model.",
        )
        return psf / np.sum(psf)
