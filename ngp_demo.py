"""
Demo file to reconstruct the entile FOV with a single InstantNGP network.
"""

import os
import random
import warnings
import time
import yaml

import numpy as np
import tifffile as tiff
from tqdm import tqdm

import tinycudann as tcnn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import misc.visualize as vis
import misc.loading_data as ld
from misc.utils import *
from misc.models import *
from misc.losses import *
from misc.propagator import PhysicalPropagator as PP
from misc.physics_params import PhysicsParamsManager
from opt import init_opts, save_opts
from fluor_rec3d import Logger, WarmupCosineAnnealingLR


def load_config(file: str):
    assert os.path.exists(file), "File not found: {}".format(file)
    assert file.endswith("yml") or file.endswith(
        "yaml"
    ), f"Should load config from yaml file, but given .{file.split('.')[-1]} file"
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config


class HashGridNeRF(nn.Module):
    def __init__(self, hash_grid_config, nerf_hid_dim, nerf_hid_layer_num, dtype=None) -> None:
        super().__init__()

        self.hash_grid_config = hash_grid_config
        self.mlp_config = (
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": nerf_hid_dim,
                "n_hidden_layers": nerf_hid_layer_num,
            }
            if nerf_hid_dim <= 128
            else {
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": nerf_hid_dim,
                "n_hidden_layers": nerf_hid_layer_num,
            }
        )

        self.dtype = dtype
        self.hash_grid_encoding = tcnn.Encoding(
            3, self.hash_grid_config, dtype=dtype, seed=random.randint(0, 1524367)
        )
        self.mlp = tcnn.Network(
            self.hash_grid_encoding.n_output_dims,
            1,
            self.mlp_config,
            seed=random.randint(0, 1524367),
        )

    def forward(self, coords):
        orig_shape = coords.shape

        encoded_pos = self.hash_grid_encoding(coords.reshape(-1, 3))

        density = self.mlp(encoded_pos).reshape(*orig_shape[:-1], -1)

        density = density.float()

        return density


data_fidelity_fun = {"mse": mse_loss, "mae": mae_loss, "ssim": ssim_loss}

if __name__ == "__main__":
    from misc.models import input_coord_3d

    # title: Basic arguments
    warnings.filterwarnings("ignore")
    args = init_opts()
    DEVICE = torch.device(f"cuda:{args.gpu_list[0]}")
    DTYPE = torch.cuda.FloatTensor
    args.psf_path = os.path.join(args.root_dir, args.psf_name)
    args.ref_path = os.path.join(args.root_dir, args.ref_name)
    args.pretrain_loss = "hybrid"
    projection_type = args.projection_type
    args.save_dir = os.path.join(args.exp_dir, args.exp_name)
    args.step_image_dir = os.path.join(args.save_dir, "step_images")
    args.blocks_dir = os.path.join(args.save_dir, "blocks")
    physics_manager = PhysicsParamsManager(args)

    pretrained_model_name = (
        f"Pretrained_InstantNGP_{args.root_dir.split('/')[-1]}_Iter{args.pretraining_num_iter}.pth"
    )
    pretrain_model_store_path = os.path.join(
        args.net_obj_save_path_trained_prefix, pretrained_model_name
    )
    trained_model_name = (
        f"Trained_InstantNGP_{args.root_dir.split('/')[-1]}_Iter{args.training_num_iter}.pth"
    )
    train_model_store_path = os.path.join(args.net_obj_save_path_trained_prefix, trained_model_name)

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.step_image_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.blocks_dir, exist_ok=True)
    save_opts(args, args.save_dir)
    save_step_list = [100 * i for i in range(6)] + [500 * i for i in range(2, 20)]

    whole_FOV = tiff.imread(os.path.join(args.root_dir, args.data_stack_name))
    whole_FOV = whole_FOV.astype(np.float32) / whole_FOV.max()
    init_FOV = whole_FOV.copy()

    if os.path.exists(os.path.join(args.root_dir, args.ref_name)):
        whole_ref = tiff.imread(os.path.join(args.root_dir, args.ref_name))
        whole_ref = whole_ref.astype(np.float32) / whole_ref.max()
        # recon_metric = reconstruction_metric(DEVICE)
    else:
        whole_ref = None

    psf_shape = args.psf_shape

    start_time = time.time()

    # title: Loading datas
    Dataloader = ld.MicroDataLoader(
        args,
        physics_manager.PSF_params(),
        measurement=whole_FOV,
        init_guess=whole_FOV,
        ref=whole_ref,
    )
    data_pack = Dataloader.load(
        invert_ref=False,
        invert_sample=False,
        psf_constraint=False,
        PSF_shape=psf_shape,
    )
    psf = torch.from_numpy(data_pack["psf"]).type(DTYPE).to(DEVICE)
    y = torch.from_numpy(data_pack["y"]).type(DTYPE).to(DEVICE)
    init_data = torch.from_numpy(data_pack["init_data"]).type(DTYPE).to(DEVICE)
    if "ref" in data_pack.keys():
        ref = torch.from_numpy(data_pack["ref"]).type(DTYPE).to(DEVICE)
        ref_exist = True
    else:
        ref_exist = False
    y_max = data_pack["y_max"]
    y_min = data_pack["y_min"]
    init_max = data_pack["init_max"]
    INPUT_DEPTH, INPUT_WIDTH, INPUT_HEIGHT = data_pack["y_shape"]
    res = (args.psf_dz, args.psf_dx, args.psf_dy)
    recover_shape = (INPUT_DEPTH, INPUT_WIDTH, INPUT_HEIGHT)

    # title: Construct models
    config = load_config(os.path.join(args.root_dir, "ngp.yaml"))
    net = HashGridNeRF(config, args.nerf_num_filters, args.nerf_num_layers).to(DEVICE)
    space_region = input_coord_3d(*recover_shape).to(DEVICE)

    propagator = PP(
        psf, psf_shape=psf_shape, obj_shape=recover_shape, boundary=args.boundary_holding_mode
    )
    win_params = physics_manager.WIN_params(recover_shape)
    fdmae_loss = GFDMAE_Loss(recover_shape, win_params, device=DEVICE)

    # title: Pretraining
    try:
        net.load_state_dict(torch.load(pretrain_model_store_path))
        print("Pre-trained model loaded.")
    except Exception as e:
        print(f"{e}\n Error occurs when loading pretrained model")
        print("Start Pretraining")
        lr = args.pretraining_lr
        num_iter = args.pretraining_num_iter
        optimizer = torch.optim.Adam(
            [{"params": net.parameters(), "lr": lr}],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = CosineAnnealingLR(optimizer, num_iter, lr / 25)
        loss_list = np.empty(shape=(1 + num_iter,))
        loss_list[:] = np.NaN
        tbar = tqdm(
            range(num_iter + 1),
        )
        hybrid_loss = lambda x, y: (ssim_loss(x, y) + fdmae_loss(x, y)) / 2
        pretrain_loss = {"hybrid": hybrid_loss, "mse": mse_loss, "ssim_loss": ssim_loss}[
            args.pretrain_loss
        ]

        for step in tbar:
            obj = net(space_region)
            obj = torch.reshape(obj, recover_shape)
            loss = pretrain_loss(obj, init_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list[step] = loss.item()
            out_x_max = obj.max().detach().cpu().numpy()
            out_x_min = obj.min().detach().cpu().numpy()

            tbar.set_description(
                "Loss: %.6f|x_max: %.6f|x_min: %.6f" % (loss.item(), out_x_max, out_x_min)
            )
            tbar.refresh()

        fit_y_np = obj.detach().cpu().numpy()
        vis.display_stack_MIP(fit_y_np, "Pretrain Fit", projection_type, "gray", args.save_dir)
        vis.display_curve(
            -10 * np.log10(loss_list), "-log10 of Pretrain Loss", args.save_dir, "loss"
        )

        fit_y_16b = fit_y_np / fit_y_np.max() * 65535
        fit_y_16b = fit_y_16b.astype(np.uint16)
        tiff.imsave(
            os.path.join(args.save_dir, "pretrain_16b.tif"),
            fit_y_16b,
        )

        if args.saving_model == "True":
            torch.save(net.state_dict(), pretrain_model_store_path)
            print("Pretrained network saved to " + pretrain_model_store_path)
        else:
            print("Pretrained network not saved.")
        # del scheduler, optimizer
        # torch.cuda.empty_cache()

    # title: Training

    lr = args.training_lr_obj
    num_iter = args.training_num_iter

    tbar = tqdm(
        range(num_iter + 1),
    )

    loss_list = np.empty(shape=(1 + num_iter,))
    loss_list[:] = np.NaN
    loss_val_list = np.empty(shape=(1 + num_iter,))
    loss_val_list[:] = np.NaN

    optimizer = torch.optim.Adam(
        [{"params": net.parameters(), "lr": lr}],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        num_iter // 20,
        num_iter,
        lr / 25,
    )
    for step in tbar:
        obj = net(space_region)
        obj = torch.reshape(obj, recover_shape) * args.nerf_max_val
        meas_pred = propagator.propagate(obj)
        pred = (meas_pred + y_min) / y_max
        data_loss = (
            data_fidelity_fun[args.data_fidelity_term](pred, y)
            + fdmae_loss(pred, y) * args.fdmae_loss_weight
        )
        reg_loss = hessian_loss(obj, args.hessian_z_scale) * args.hessian_weight
        loss = data_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # title: Logging
        with torch.no_grad():
            if step in save_step_list:
                Logger.get_instance().dispaly_MIP(
                    obj.detach().cpu().numpy(),
                    "network_output,step:%d" % step,
                    projection_type,
                    "gray",
                    args.step_image_dir,
                )
                Logger.get_instance().dispaly_MIP(
                    meas_pred.detach().cpu().numpy(),
                    "conv_output,step:%d" % step,
                    projection_type,
                    "gray",
                    args.step_image_dir,
                )
            loss_list[step] = data_loss.item()
            obj_max = torch.max(obj).item()
            obj_min = torch.min(obj).item()
            pred_max = torch.max(meas_pred).item()
            pred_min = torch.min(meas_pred).item()
            if ref_exist:
                val_loss = torch.mean(torch.square(obj / obj.max() - ref)).item()
                loss_val_list[step] = val_loss
                tbar_str = "loss=%.3e|data=%.3e|max_x=%.3e|min_x=%.3e|val_psnr=%.2fdB" % (
                    loss.item(),
                    data_loss.item(),
                    obj_max,
                    obj_min,
                    -10 * np.log10(val_loss),
                )
            else:
                tbar_str = "loss=%.3e|data=%.3e|xx=%.1f|nx=%.1e|xy=%.2f|ny=%.1e" % (
                    loss.item(),
                    data_loss.item(),
                    obj_max,
                    obj_min,
                    pred_max,
                    pred_min,
                )
        tbar.set_description(tbar_str)
        tbar.refresh()

    vis.display_curve(-10 * np.log10(loss_list), "Training Loss Log10", args.save_dir, "loss")
    if ref_exist:
        vis.display_curve(-10 * np.log10(loss_val_list), "Validation PSNR", args.save_dir, "PSNR")
        val_loss = torch.mean(torch.square(obj / obj.max() - ref)).item()
        print("Validation PSNR: %.3fdB" % (-10 * np.log10(val_loss)))
    else:
        print("No reference data, skip validation.")
        loss = torch.mean(torch.square((meas_pred + y_min) / y_max - y)).item()
        print("PSNR comparing with measurement: %.3fdB" % (-10 * np.log10(loss)))

    print("Max value of reconstructed object: %.3e" % torch.max(obj).item())

    print("Training finished.")
    recover_obj = obj.detach().cpu().numpy() + y_min
    pred_meas = meas_pred.detach().cpu().numpy()
    full_obj = obj.detach().cpu().numpy()
    full_pred_meas = meas_pred.detach().cpu().numpy()

    y = y.detach().cpu().numpy()
    if ref_exist:
        ref = ref.detach().cpu().numpy()
        vis.display_multistack_MIP(
            [y, pred_meas, recover_obj, ref],
            "meas-pred_meas-obj-ref",
            projection_type,
            "gray",
            args.save_dir,
        )
        # print the PSNR between the recovered object and the reference object
        psnr = -10 * np.log10(np.mean(np.square(recover_obj / recover_obj.max() - ref)))
        print("Validation PSNR: %.3fdB" % (psnr))
    else:
        vis.display_multistack_MIP(
            [y, pred_meas, recover_obj],
            "meas-pred_meas-obj",
            projection_type,
            "viridis",
            args.save_dir,
        )
    save_rec = recover_obj
    Logger.get_instance().save_stack(
        save_rec, "rec_" + args.data_stack_name.split(".")[0], args.blocks_dir
    )

    recover_obj_16b = recover_obj / recover_obj.max() * 65535
    recover_obj_16b = recover_obj_16b.astype(np.uint16)
    tiff.imsave(
        os.path.join(args.save_dir, "rec_whole_FOV_16b.tif"),
        recover_obj_16b,
    )
    print("Total time consuming: {:.2f}min".format((time.time() - start_time) / 60))
