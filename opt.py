import argparse
import os
import json
import yaml


def set_opts():
    """Set the hyperparameters.
    Add a ArgumentParser object, and add arguments to it.
    Return the ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="Hyperparameters - PIECE-INR")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file to specify optical system parameters",
    )
    # title: Working directory
    parser.add_argument("--exp_dir", type=str, default="./exp/")
    parser.add_argument(
        "--data_stack_name",
        type=str,
        default="measurement.tif",
        help="Name of the wide-field measurement stack",
    )
    parser.add_argument(
        "--init_stack_name",
        type=str,
        default="measurement.tif",
        help="Initial guess, if not specified, use name of the measurement stack",
    )
    parser.add_argument(
        "--root_dir", type=str, default="./source", help="Where your data files are located"
    )
    parser.add_argument("--psf_name", type=str, default="psf.tif")
    parser.add_argument("--ref_name", type=str, default="sample.tif")
    parser.add_argument("--exp_name", type=str, default="test_exp")

    parser.add_argument("--net_obj_save_path_pretrained_prefix", type=str, default="./rec/")
    parser.add_argument("--net_obj_save_path_trained_prefix", type=str, default="./rec/")

    parser.add_argument(
        "--saving_model",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Save the model weights or not",
    )
    # title: Physical parameters
    parser.add_argument(
        "--psf_dz",
        type=float,
        default=0.2,
        help="Axial scanning interval on wide-field microscopy (um)",
    )  # um
    parser.add_argument(
        "--psf_dy",
        type=float,
        default=0.086,
        help="Lateral pixel size of wide-field microscopy (um)",
    )  # um
    parser.add_argument(
        "--psf_dx",
        type=float,
        default=0.086,
        help="Lateral pixel size of wide-field microscopy (um)",
    )  # um
    parser.add_argument("--NA", type=float, default=1.1, help="Numerical Aperture")
    parser.add_argument("--wl", type=float, default=0.515, help="Emission light wave length (um)")
    parser.add_argument("--ns", type=float, default=1.333, help="Refractive index of the object")
    parser.add_argument(
        "--ni", type=float, default=1, help="Refractive index of the immersion medium"
    )
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
        "--projection_type",
        type=str,
        default="max",
        choices=["max", "min"],
        help="max: maximum projection, min: minimum projection.",
    )
    # title: Block-wise control
    parser.add_argument(
        "--boundary_holding_mode",
        type=str,
        default="DEFAULT",
        choices=["DEFAULT", "ZERO", "REFLECT", "REPEAT", "REPLICATE", "NULL"],
        help="DEFAULT: default mode, ZERO: zero padding, REFLECT: reflect padding, REPEAT: repeat padding, NULL: no padding.",
    )
    parser.add_argument(
        "--axial_pad_length", type=int, default=20, help="Additional axial padding length"
    )  # 0, 10, 20; should be even number
    parser.add_argument(
        "--lateral_pad_length", type=int, default=20, help="Additional lateral padding length"
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
        help="Varianve gate for a block to be considered as a pure background block. \
            If the varianve and mean value of a block is smaller than corresponding gates, \
            it is considered as a pure background block.",
    )
    parser.add_argument(
        "--pure_background_mean_gate",
        type=float,
        default=0.0,
        help="Varianve gate for a block to be considered as a pure background block.",
    )
    # title: Positional encoding
    parser.add_argument(
        "--encoding_option",
        type=str,
        default="PIEE",
        choices=[
            "cartesian",
            "radial_cartesian",
            "gaussian",
            "spherical",
            "PIEE",
        ],
    )
    parser.add_argument("--freq_logscale", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--cartesian_encoding_dim", type=int, default=3)
    parser.add_argument("--cartesian_encoding_depth", type=int, default=6)
    parser.add_argument(
        "--gaussian_scale", type=float, default=14, help="sigma in gaussian encoding"
    )
    parser.add_argument(
        "--gaussian_encoding_num", type=int, default=256, help="number of gaussian encoding"
    )
    parser.add_argument(
        "--zenith_encoding_angle",
        type=float,
        default=45,
        help="Zenith angle step size. Typically, 45 ~ 60 for PIEE encoding.",
    )
    parser.add_argument(
        "--radial_encoding_angle",
        type=float,
        default=9,  # the angle step size in radial direction
        help="Typically, 9 ~ 15 for PIEE encoding. Smaller values indicates the ability to represent finer features.",
    )
    parser.add_argument(
        "--radial_encoding_depth",
        type=int,
        default=6,
        help="If too large, stripe artifacts. If too small, oversmoothened features. Typically, 6 or 7.",
    )
    # title: Network architecture
    parser.add_argument("--nerf_num_layers", type=int, default=4)
    parser.add_argument("--nerf_num_filters", type=int, default=128)
    parser.add_argument("--nerf_skips", type=list, default=[2, 4, 6])
    parser.add_argument("--nerf_beta", type=float, default=None)  # 1.0 or None (sigmoid)
    parser.add_argument("--nerf_max_val", type=float, default=50.0)
    # title: Different INR architectures
    parser.add_argument(
        "--inr_act_type",
        type=str,
        default="ReLU",
        choices=["ReLU", "SIREN", "WIRE", "HashGrid"],
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
    parser.add_argument(
        "--gpu_list",
        nargs="*",
        type=int,
        default=[0],
        help="Usable GPU list. Auto parallelization for multiple GPUs",
    )
    parser.add_argument(
        "--pretraining",
        type=str,
        default="True",
        choices=["True", "False"],
    )  # True, False
    parser.add_argument(
        "--loading_pretrained_model",
        type=str,
        default="True",
        choices=["True", "False"],
    )
    parser.add_argument("--log_option", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--pretraining_num_iter", type=int, default=1000)  # 2500
    parser.add_argument("--pretraining_lr", type=float, default=1e-3)
    parser.add_argument("--training_num_iter", type=int, default=2000)
    parser.add_argument("--training_lr_obj", type=float, default=1e-3)

    # title: Loss function control
    parser.add_argument(
        "--data_fidelity_term",
        type=str,
        default="mse",
        choices=["mse", "ssim", "mae"],
        help="Data fidelity term for the loss function.",
    )
    parser.add_argument(
        "--fdmae_loss_weight", type=float, default=1e-3, help="Weight for G-FDMAE loss."
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
    """
    Init hyper parameters with default values.
    Priority:
        args.config > shell command > default argument
    """
    parser = set_opts()
    args = parser.parse_args()
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
        print(f"Safely loading pre-set params from {args.config}")
    else:
        print(f"{args.config} not exists, using input params")
    return args


def load_opts(file_path, args: argparse.Namespace = None):
    """Load hyper parameters from a json or yaml file."""
    assert os.path.exists(file_path), "File not found: {}".format(file_path)
    if args is None:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
    with open(file_path, "r") as f:
        if file_path.endswith("yml") or file_path.endswith("yaml"):
            args.__dict__ = yaml.safe_load(f, indent=2)
        elif file_path.endswith("txt") or file_path.endswith("json"):
            args.__dict__ = json.load(f, indent=2)
    return args


def save_opts(args, file_dir, type="yaml"):
    if type == "json":
        file_name = os.path.join(file_dir, "args.txt")
        with open(file_name, "w") as f:
            json.dump(args.__dict__, f, indent=2)
    elif type == "yaml":
        file_name = os.path.join(file_dir, "args.yml")
        with open(file_name, "w") as f:
            yaml.dump(args.__dict__, f, indent=2)
    else:
        raise NotImplementedError(f"Unsupported file type {type}")

    return None


if __name__ == "__main__":
    parser = set_opts()
    parser.print_help()
