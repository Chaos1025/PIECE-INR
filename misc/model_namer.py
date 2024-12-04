import os


def spherical_encoding_suffixs(args, iter):
    suffix = ("_" + args.data_stack_name.split(".")[0]                  # data name
                + "_" + "PosEncode" + "_" + args.encoding_option        # encoding option
                + "_" + "ZencLit" + str(args.zenith_encoding_angle)     # zenith encoding angle
                + "_" + "ZencNum" + str(args.cartesian_encoding_depth)  # zenith encoding depth
                + "_" + "AencLit"  + str(args.radial_encoding_angle)    # radial encoding angle
                + "_" + "RencNum" + str(args.radial_encoding_depth)     # radial encoding depth
                + "_" + "NetDepth" + str(args.nerf_num_layers)          # network depth
                + "_" + "NetWidth" + str(args.nerf_num_filters)         # network width
                + "_" + "Rpad" + str(args.lateral_pad_length)           # lateral padding length
                + "_" + "Zpad" + str(args.axial_pad_length)             # axial padding length
                + "_" + "Beta" + str(args.nerf_beta)                    # last activation type
                + "_" + "Maxval" + str(args.nerf_max_val)               # output range
                + "_" + "Iter" + str(iter)         # number of iterations
                + ".pth")
    return str(suffix)

def gaussian_encoding_suffixs(args, iter):
    suffix = ("_" + args.data_stack_name.split(".")[0]                  # data name
                + "_" + "PosEncode" + "_" + args.encoding_option        # encoding option
                + "_" + "Scale" + "_" + str(args.gaussian_scale)             # gaussian scale
                + "_" + "GaussianNum" + str(args.gaussian_encoding_num)  # gaussian encoding depth
                + "_" + "NetDepth" + str(args.nerf_num_layers)          # network depth
                + "_" + "NetWidth" + str(args.nerf_num_filters)         # network width
                + "_" + "Rpad" + str(args.lateral_pad_length)           # lateral padding length
                + "_" + "Zpad" + str(args.axial_pad_length)             # axial padding length
                + "_" + "Beta" + str(args.nerf_beta)                    # last activation type
                + "_" + "Maxval" + str(args.nerf_max_val)               # output range
                + "_" + "Iter" + str(iter)         # number of iterations
                + ".pth")
    return str(suffix)

def positional_encoding_suffixs(args, iter):
    suffix = ("_" + args.data_stack_name.split(".")[0]                  # data name
                + "_" + "PosEncode" + "_" + args.encoding_option        # encoding option
                + "_" + "ZencNum" + str(args.cartesian_encoding_depth)  # zenith encoding depth
                + "_" + "NetDepth" + str(args.nerf_num_layers)          # network depth
                + "_" + "NetWidth" + str(args.nerf_num_filters)         # network width
                + "_" + "Rpad" + str(args.lateral_pad_length)           # lateral padding length
                + "_" + "Zpad" + str(args.axial_pad_length)             # axial padding length
                + "_" + "Beta" + str(args.nerf_beta)                    # last activation type
                + "_" + "Maxval" + str(args.nerf_max_val)               # output range
                + "_" + "Iter" + str(iter)         # number of iterations
                + ".pth")
    return str(suffix)

def radial_encoding_suffixs(args, iter):
    suffix = ("_" + args.data_stack_name.split(".")[0] # data name
                + "_" + "PosEncode" + "_" + args.encoding_option        # encoding option
                + "_" + "ZencNum" + str(args.cartesian_encoding_depth)  # zenith encoding depth
                + "_" + "AencLit"  + str(args.radial_encoding_angle)    # radial encoding angle
                + "_" + "RencNum" + str(args.radial_encoding_depth)     # radial encoding depth
                + "_" + "NetDepth" + str(args.nerf_num_layers)          # network depth
                + "_" + "NetWidth" + str(args.nerf_num_filters)         # network width
                + "_" + "Rpad" + str(args.lateral_pad_length)           # lateral padding length
                + "_" + "Zpad" + str(args.axial_pad_length)             # axial padding length
                + "_" + "Beta" + str(args.nerf_beta)                    # last activation type
                + "_" + "Maxval" + str(args.nerf_max_val)               # output range
                + "_" + "Iter" + str(iter)         # number of iterations
                + ".pth")
    return str(suffix)

def inr_suffixs(args, iter):
    suffix = ("_" + args.data_stack_name.split(".")[0] 
                + "_" + "INRAct" + "_" + args.inr_act_type
                + "_" + "NetDepth" + str(args.nerf_num_layers)          # network depth
                + "_" + "NetWidth" + str(args.nerf_num_filters)         # network width
                + "_" + "Beta" + str(args.nerf_beta)                    # last activation type
                + "_" + "Maxval" + str(args.nerf_max_val)               # output range
                + "_" + "Iter" + str(iter)
                + ".pth")
    return str(suffix)


def pretrain_model_path(args, model_name=""):
    if args.inr_act_type in ["SIREN", "Gauss", "WIRE", ]:
        pretrain_suffix = inr_suffixs(args, args.pretraining_num_iter)
    elif args.encoding_option == "spherical" or args.encoding_option == "PISE":
        pretrain_suffix = spherical_encoding_suffixs(args, args.pretraining_num_iter)
    elif args.encoding_option == "gaussian":
        pretrain_suffix = gaussian_encoding_suffixs(args, args.pretraining_num_iter)
    elif args.encoding_option == "radial_cartesian":
        pretrain_suffix = radial_encoding_suffixs(args, args.pretraining_num_iter)
    else: # cartesian embedding
        pretrain_suffix = positional_encoding_suffixs(args, args.pretraining_num_iter)
    
    if model_name:
        model_name = model_name + "_"

    pretrain_model_path = os.path.join(
        args.net_obj_save_path_pretrained_prefix,
        "PretrainModel_" + model_name + args.root_dir.split("/")[-2] + 
        pretrain_suffix,
    )
    
    return pretrain_model_path

def trained_model_path(args, model_name=""):
    if args.inr_act_type in ["SIREN", "Gauss", "WIRE", ]:
        trained_suffix = inr_suffixs(args, args.training_num_iter)
    elif args.encoding_option == "spherical" or args.encoding_option == "PISE":
        trained_suffix = spherical_encoding_suffixs(args, args.training_num_iter)
    elif args.encoding_option == "gaussian":
        trained_suffix = gaussian_encoding_suffixs(args, args.training_num_iter)
    elif args.encoding_option == "radial_cartesian":
        trained_suffix = radial_encoding_suffixs(args, args.training_num_iter)
    else: # cartesian embedding
        trained_suffix = positional_encoding_suffixs(args, args.training_num_iter)
    
    if model_name:
        model_name = model_name + "_"

    trained_model_path = os.path.join(
        args.net_obj_save_path_trained_prefix,
        "TrainedModel_" + model_name + args.root_dir.split("/")[-2] + 
        trained_suffix,
    )
    
    return trained_model_path