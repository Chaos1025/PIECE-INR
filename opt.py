import argparse
import os
import json


def set_opts():
    """Set the hyperparameters.
    Add a ArgumentParser object, and add arguments to it.
    Return the ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="Hyperparameters - Beads")
    # title: Working directory
    parser.add_argument("--source_dir", type=str, default="./source/")
    parser.add_argument("--exp_dir", type=str, default="./exp/")
    parser.add_argument("--data_stack_name", type=str, default="FITC-square.tif")
    parser.add_argument("--init_stack_name", type=str, default="FITC-square.tif")
    parser.add_argument("--rl_stack_name", type=str, default="RL-FITC-square.tif")
    parser.add_argument("--root_dir", type=str, default="./source/elegans/FITC")
    parser.add_argument("--psf_name", type=str, default="FITC_PSF_crop2.tif")
    parser.add_argument("--ref_name", type=str, default="GT_crop2.tif")
    parser.add_argument("--exp_name", type=str, default="elegans-FITC")

    parser.add_argument(
        "--net_obj_save_path_pretrained_prefix", type=str, default="./rec/"
    )
    parser.add_argument(
        "--net_obj_save_path_trained_prefix", type=str, default="./rec/"
    )

    parser.add_argument("--normalized", type=bool, default=False)
    parser.add_argument(
        "--saving_model", type=str, default="True", choices=["True", "False"]
    )
    # title: Physical parameters
    parser.add_argument("--psf_dz", type=float, default=0.2)  # um
    parser.add_argument("--psf_dy", type=float, default=0.086) # um
    parser.add_argument("--psf_dx", type=float, default=0.086) # um
    parser.add_argument("--n_detection", type=float, default=1.1, help="Numerical Aperture")
    parser.add_argument("--emission_wavelength", type=float, default=0.515, help="Emission light wave length (um)")
    parser.add_argument("--n_obj", type=float, default=1.333, help="Refractive index of the object")
    parser.add_argument("--n_immersion", type=float, default=1, help="Refractive index of the immersion medium")
    parser.add_argument(
        "--psf_generation",
        type=str,
        default="Angular",
        choices=["Gibson", "Angular", "External"],
        help="Angular: angular spectrum method, Gibson: Gibson-Lanni PSF, External: load PSF from file.",
    )
    parser.add_argument(
        "--psf_shape",
        nargs="+",
        type=int,
        default=[64, 128, 128],
        help="Shape of the PSF, [dim_d, dim_h, dim_w]",
    )
    parser.add_argument(
        "--working_type",
        type=str,
        default="solver",
        choices=["solver", "simulator"],
        help="solver: solving deconvolution with given measurement, simulator: make simulation for deconvolution.",
    )
    parser.add_argument(
        "--projection_type",
        type=str,
        default="max",
        choices=["max", "min"],
        help="max: maximum projection, min: minimum projection.",
    )
    # title: Block-wise control
    parser.add_argument(
        "--axial_pad_length", type=int, default=20
    )  # 0, 10, 20; should be even number
    parser.add_argument(
        "--lateral_pad_length", type=int, default=20
    )  # 0, 20, 40; should be even number
    parser.add_argument(
        "--lateral_view",
        type=int,
        default=100,
        help="View size of a single block in lateral direction",
    )
    parser.add_argument(
        "--lateral_overlap",
        type=int,
        default=20,
        help="Overlap size between adjacent blocks in lateral direction",
    )
    parser.add_argument(
        "--axial_view",
        type=int,
        default=120,
        help="View size of a single block in axial direction",
    )
    parser.add_argument(
        "--row_picker", nargs="*", type=int, help="Row picker for the blocks", default=[]
    )
    parser.add_argument(
        "--col_picker", nargs="*", type=int, help="Col picker for the blocks", default=[]
    )
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="smooth",
        choices=["smooth", "steep"],
        help="smooth: smooth mask, steep: steep mask.",
    )
    parser.add_argument(
        "--pure_background_variance_gate",
        type=float,
        default=0.0,
        help="Varianve gate for a block to be considered as a pure background block. Both the var and mean gates should be satisfied. \
            If the varianve and mean value of a block is smaller than corresponding gates, it is considered as a pure background block.",
    )
    parser.add_argument(
        "--pure_background_mean_gate",
        type=float,
        default=0.0,
        help="Varianve gate for a block to be considered as a pure background block. Both the var and mean gates should be satisfied.",
    )
    # title: Positional encoding
    parser.add_argument(
        "--encoding_option",
        type=str,
        default="PISE",
        choices=[
            "cartesian",
            "radial",
            "tri_radial",
            "radial_cartesian",
            "gaussian",
            "spherical",
            "PISE",
            # "PISE_1",
            # "PISE_2",
        ],
    )
    parser.add_argument("--freq_logscale", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--cartesian_encoding_dim", type=int, default=3)
    parser.add_argument("--cartesian_encoding_depth", type=int, default=8)
    parser.add_argument(
        "--gaussian_scale", 
        type=float, 
        default=26, 
        help="sigma in gaussian encoding"
    )
    parser.add_argument(
        "--gaussian_encoding_num", 
        type=int, 
        default=256, 
        help="number of gaussian encoding"
    )
    parser.add_argument(
        "--zenith_encoding_angle",
        type=float,
        default=45,  # the angle step size in zenith direction
        help="the angle step size of zenith angle",
    )
    parser.add_argument(
        "--radial_encoding_angle",
        type=float,
        default=9,  # the angle step size in radial direction
        help="Typically, 3 ~ 7.5. Smaller values indicates the ability to represent fine features.",
    )
    parser.add_argument(
        "--radial_encoding_depth",
        type=int,
        default=6,
        help="If too large, stripe artifacts. If too small, oversmoothened features. Typically, 6 or 7.",
    )  # 7, 8 (jiggling artifacts)
    # title: Network architecture
    parser.add_argument("--nerf_num_layers", type=int, default=6)
    parser.add_argument(
        "--nerf_num_filters", type=int, default=128
    )  # 32 (not enough), 64, 128 / at least y_.shape[0]/2? Helps to reduce artifacts fitted to aberrated features and noise.
    parser.add_argument(
        "--nerf_skips", type=list, default=[2, 4, 6]
    )  # [2,4,6], [2,4,6,8]: good, [2,4], [4], [4, 8]: insufficient.
    parser.add_argument(
        "--nerf_beta", type=float, default=None
    )  # 1.0 or None (sigmoid)
    parser.add_argument("--nerf_max_val", type=float, default=10.0)
    # title: Different INR architectures
    parser.add_argument(
        "--inr_act_type",
        type=str,
        default="ReLU",
        choices=["ReLU", "SIREN", "Gauss", "WIRE"],
        help="Choose activation function for the network. Determines the network architecture. Prior to 'encoding_option'.",
    )
    parser.add_argument(
        "--inr_num_layers",
        type=int,
        default=1,
        help="Layer number for the network like SIREN, Gauss, WIRE that use different activation functions.",
    )
    parser.add_argument(
        "--inr_num_filters",
        type=int,
        default=256,
        help="Filter number for the network like SIREN, Gauss, WIRE that use different activation functions.",
    )
    # title: Training
    parser.add_argument("--gpu_list", nargs="*", type=int, default=[0], help="Usable GPU list")
    parser.add_argument(
        "--lr_schedule", type=str, default="cosine", choices=["multi_step", "cosine"]
    )
    parser.add_argument("--pretraining", type=bool, default=True)  # True, False
    parser.add_argument(
        "--loading_pretrained_model",
        type=str,
        default="True",
        choices=["True", "False"],
    )
    parser.add_argument(
        "--log_option", type=str, default="True", choices=["True", "False"]
    )
    parser.add_argument("--pretraining_num_iter", type=int, default=1000)  # 2500
    parser.add_argument("--pretraining_lr", type=float, default=1e-3)
    parser.add_argument(
        "--training_num_iter", type=int, default=2000
    )  # 100 for donut, 300 for bead
    parser.add_argument("--training_lr_obj", type=float, default=1e-3)
    parser.add_argument("--training_lr_hash", type=float, default=1e-4)
    parser.add_argument("--training_lr_ker", type=float, default=1e-3)  # 1e-2
    parser.add_argument("--kernel_max_val", type=float, default=1e-2)
    parser.add_argument("--kernel_order_up_to", type=int, default=4)  # True, False

    # title: Loss function control
    parser.add_argument(
        "--data_fidelity_term", type=str, default="mse", choices=["mse", "ssim", "mae"]
    )
    parser.add_argument("--fdmae_loss_weight", type=float, default=1e-3)
    parser.add_argument("--rl_loss_weight", type=float, default=1e-2)
    parser.add_argument("--l1_weight", type=float, default=0)
    parser.add_argument(
        "--tv_loss_weight",
        nargs="+",
        type=float,
        default=[0, 0, 0],
        help="tv_loss_weight = [tv_z, tv_x, tv_y]",
    )
    parser.add_argument(
        "--hessian_weight",
        type=float,
        default=0,
        help="Weight params for hessian regularization.",
    )
    parser.add_argument(
        "--hessian_z_scale",
        type=float,
        default=1.0,
        help="Scale param along z axis in hessian regularization.",
    )

    return parser


def init_opts():
    """Init hyper parameters with default values."""
    parser = set_opts()
    args = parser.parse_args()
    return args


# args = parser.parse_args(args=[])


def load_opts(file_path):
    """Load hyper parameters from a json file."""
    assert os.path.exists(file_path), "File not found: {}".format(file_path)
    assert file_path.endswith(".txt"), "Only support .txt file."
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(file_path, "r") as f:
        args.__dict__ = json.load(f)
    return args
