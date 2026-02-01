"""File to handle data loading."""

import torch
import os
import numpy as np
import tifffile as tiff

import microscPSF.microscPSF as msPSF
from scipy.io import loadmat

dtype = torch.cuda.FloatTensor


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
        if diff > 0:  # If mat's dimension is larger, trim it, tail first
            trim_size = diff // 2
            mat = np.take(mat, range(trim_size, trim_size + shape[i]), axis=i)
        elif diff < 0:  # If mat's dimension is smaller, pad it, head first
            pad_size = int(np.ceil(abs(diff) / 2))
            pad_width = [
                (0, 0) if j != i else (pad_size, pad_size - diff % 2) for j in range(len(shape))
            ]
            mat = np.pad(mat, pad_width, mode="constant")
    return mat


class MicroDataLoader:
    def __init__(
        self,
        args,
        phys_params,
        measurement=None,
        init_guess=None,
        psf=None,
        ref=None,
    ):
        assert args.psf_generation in ["Gibson", "Angular", "External"]
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

        self.measurement = measurement
        self.init_data = init_guess
        self.psf = psf
        self.ref = ref

    def load_data(self, data_dir):
        # loading corresponding .tif data
        data = tiff.imread(data_dir).astype(np.float32)
        min_data = data.min()
        max_data = data.max()
        data = (data - min_data) / (
            max_data - min_data
        )  # Subtract minimum value (constant background) and normalize sample
        # print("Normalize data from [{},{}] to [0,1]".format(min_data, max_data))
        return data

    def load_external_psf(self, psf_shape, psf_dir: str) -> np.ndarray:
        # Loading PSF from external file, and adjust its shape if necessary
        if psf_dir.endswith(".tif"):
            psf_ = tiff.imread(psf_dir).astype(np.float32)
        elif psf_dir.endswith(".mat"):
            psf_ = loadmat(psf_dir)["psf"]
            # Matlab use a different sequence of axis
            psf_ = np.transpose(psf_, (2, 1, 0)).astype(np.float32)
        # print("Load PSF from external file with shape of: ", psf_.shape)
        if psf_shape is not None and psf_.shape != psf_shape:
            # if shape diff is even, adjust shape equally
            # if odd, adjust like [N//2, N//2+1]
            psf_ = adjust_matrix_shape(psf_, psf_shape)
        # psf_ = psf_ / np.sum(psf_, axis=(1,2), keepdims=True)
        psf_ = psf_ / np.sum(psf_)  # Normalizing the psf for conservation of energy
        return psf_

    def load(self, PSF_shape=None, *args, **kwargs) -> dict:
        """Load wide-field data from tiff file or variable.

        Returns:
            ret_pack: a dict containing the following keys:
                ref: the reference object, a 3d numpy array if exists.
                psf: the psf after normalization of total energy, a 3d numpy array.
                y: the measurement after normalization of max value, a 3d numpy array.

        """
        ret_pack = {}

        if self.measurement is None:
            y_ = self.load_data(self.data_dir)
        else:
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
        elif self.psf_type == "External":
            psf_ = self.load_external_psf(PSF_shape, self.psf_dir)
        else:
            raise NotImplementedError(f"PSF type {self.psf_type} has not been implemented.")

        if os.path.exists(self.ref_dir) and self.ref is None:
            ret_pack["ref"] = self.load_data(self.ref_dir)
        elif self.ref is not None:
            ret_pack["ref"] = self.ref

        ret_pack["psf"] = psf_ / psf_.sum()
        ret_pack["y"] = y_ / y_max
        ret_pack["y_max"] = y_max
        ret_pack["y_min"] = y_min * 0.95
        ret_pack["init_max"] = self.init_data.max()
        ret_pack["init_data"] = self.init_data / ret_pack["init_max"]
        ret_pack["y_shape"] = sample_shape

        # print("PSF shape after loading: ", psf_.shape)
        # print("Measurement shape after loading: ", y_.shape)
        # print("Maximum value of measurement: ", y_max)
        # print("Minimum value of measurement: ", y_min)

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
        # print(
        #     "Generate PSF with shape: ",
        #     psf.shape,
        #     " and units: ",
        #     self.units,
        #     ", using Gibson-Lanni model.",
        # )
        return psf / np.sum(psf)
