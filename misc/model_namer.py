"""
Customized functions to create specific name for our models
"""

import os


def spherical_encoding_suffixs(args, iter):
    data_name = args.data_stack_name.split(".")[0]
    suffix = (
        f"_{data_name}_PosEncode_{args.encoding_option}"  # encoding option
        + f"_ZencLit{args.zenith_encoding_angle:.0f}"  # zenith encoding angle
        + f"_ZencNum{args.cartesian_encoding_depth:.0f}"  # zenith encoding depth
        + f"_AencLit{args.radial_encoding_angle:.0f}"  # radial encoding depth
        + f"_RencNum{args.radial_encoding_depth:.0f}"  # radial encoding depth
        + f"_NetDepth{args.nerf_num_layers:.0f}"  # network depth
        + f"_NetWidth{args.nerf_num_filters:.0f}"  # network width
        + f"_Rpad{args.lateral_pad_length:.0f}"  # lateral padding length
        + f"_Zpad{args.axial_pad_length:.0f}"  # axial padding length
        + f"_Beta{args.nerf_beta}"  # last activation type
        + f"_Maxval{args.nerf_max_val:.0f}"  # output range
        + f"_Iter{iter:.0f}.pth"  # number of iterations
    )
    return str(suffix)


def gaussian_encoding_suffixs(args, iter):
    data_name = args.data_stack_name.split(".")[0]
    suffix = (
        f"_{data_name}_PosEncode_{args.encoding_option}"  # encoding option
        + f"_Scale{args.gaussian_scale:.0f}"  # gaussian scale
        + f"_GaussianNum{args.gaussian_encoding_num:.0f}"  # gaussian encoding depth
        + f"_NetDepth{args.nerf_num_layers:.0f}"  # network depth
        + f"_NetWidth{args.nerf_num_filters:.0f}"  # network width
        + f"_Rpad{args.lateral_pad_length:.0f}"  # lateral padding length
        + f"_Zpad{args.axial_pad_length:.0f}"  # axial padding length
        + f"_Beta{args.nerf_beta}"  # last activation type
        + f"_Maxval{args.nerf_max_val:.0f}"  # output range
        + f"_Iter{iter:.0f}.pth"  # number of iterations
    )
    return str(suffix)


def positional_encoding_suffixs(args, iter):
    data_name = args.data_stack_name.split(".")[0]
    suffix = (
        f"_{data_name}_PosEncode_{args.encoding_option}"  # encoding option
        + f"_ZencNum{args.cartesian_encoding_depth:.0f}"  # zenith encoding depth
        + f"_NetDepth{args.nerf_num_layers:.0f}"  # network depth
        + f"_NetWidth{args.nerf_num_filters:.0f}"  # network width
        + f"_Rpad{args.lateral_pad_length:.0f}"  # lateral padding length
        + f"_Zpad{args.axial_pad_length:.0f}"  # axial padding length
        + f"_Beta{args.nerf_beta}"  # last activation type
        + f"_Maxval{args.nerf_max_val:.0f}"  # output range
        + f"_Iter{iter:.0f}.pth"  # number of iterations
    )
    return str(suffix)


def radial_encoding_suffixs(args, iter):
    data_name = args.data_stack_name.split(".")[0]
    suffix = (
        f"_{data_name}_PosEncode_{args.encoding_option}"  # encoding option
        + f"_ZencNum{args.cartesian_encoding_depth:.0f}"  # zenith encoding depth
        + f"_AencLit{args.radial_encoding_angle:.0f}"  # radial encoding angle
        + f"_RencNum{args.radial_encoding_depth:.0f}"  # radial encoding depth
        + f"_NetDepth{args.nerf_num_layers:.0f}"  # network depth
        + f"_NetWidth{args.nerf_num_filters:.0f}"  # network width
        + f"_Rpad{args.lateral_pad_length:.0f}"  # lateral padding length
        + f"_Zpad{args.axial_pad_length:.0f}"  # axial padding length
        + f"_Beta{args.nerf_beta}"  # last activation type
        + f"_Maxval{args.nerf_max_val:.0f}"  # output range
        + f"_Iter{iter:.0f}.pth"  # number of iterations
    )
    return str(suffix)


def inr_suffixs(args, iter):
    data_name = args.data_stack_name.split(".")[0]
    suffix = (
        f"_{data_name}_INRAct_{args.inr_act_type}"  # INR activation type
        + f"_NetDepth{args.inr_num_layers:.0f}"  # network depth
        + f"_NetWidth{args.inr_num_filters:.0f}"  # network width
        + f"_Rpad{args.lateral_pad_length:.0f}"  # lateral padding length
        + f"_Zpad{args.axial_pad_length:.0f}"  # axial padding length
        + f"_Beta{args.nerf_beta}"  # last activation type
        + f"_Maxval{args.nerf_max_val:.0f}"  # output range
        + f"_Iter{iter:.0f}.pth"  # number of iterations
    )
    return str(suffix)


def pretrain_model_path(args, model_name=""):
    if args.inr_act_type in ["SIREN", "Gauss", "WIRE", "HashGrid"]:
        pretrain_suffix = inr_suffixs(args, args.pretraining_num_iter)
    elif args.encoding_option == "spherical" or args.encoding_option == "PIEE":
        pretrain_suffix = spherical_encoding_suffixs(args, args.pretraining_num_iter)
    elif args.encoding_option == "gaussian":
        pretrain_suffix = gaussian_encoding_suffixs(args, args.pretraining_num_iter)
    elif args.encoding_option == "radial_cartesian":
        pretrain_suffix = radial_encoding_suffixs(args, args.pretraining_num_iter)
    else:  # cartesian embedding
        pretrain_suffix = positional_encoding_suffixs(args, args.pretraining_num_iter)

    if model_name:
        model_name = model_name + "_"

    pretrain_model_path = os.path.join(
        args.net_obj_save_path_pretrained_prefix,
        "PretrainModel_" + model_name + args.root_dir.split("/")[-2] + pretrain_suffix,
    )

    return pretrain_model_path


def trained_model_path(args, model_name=""):
    if args.inr_act_type in ["SIREN", "Gauss", "WIRE"]:
        trained_suffix = inr_suffixs(args, args.training_num_iter)
    elif args.encoding_option == "spherical" or args.encoding_option == "PIEE":
        trained_suffix = spherical_encoding_suffixs(args, args.training_num_iter)
    elif args.encoding_option == "gaussian":
        trained_suffix = gaussian_encoding_suffixs(args, args.training_num_iter)
    elif args.encoding_option == "radial_cartesian":
        trained_suffix = radial_encoding_suffixs(args, args.training_num_iter)
    else:  # cartesian embedding
        trained_suffix = positional_encoding_suffixs(args, args.training_num_iter)

    if model_name:
        model_name = model_name + "_"

    trained_model_path = os.path.join(
        args.net_obj_save_path_trained_prefix,
        "TrainedModel_" + model_name + args.root_dir.split("/")[-2] + trained_suffix,
    )

    return trained_model_path
