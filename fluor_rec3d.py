# Author: Chenyu Xu
# Date: 2023/10/10
# Description: 3D reconstruction of wide-field fluorescence microscopy images.
#   Input:
#       - Fluorescence images stack, measured when microscope focused on different depth
#       - Point Spread Function, calculated in theory or measured in experiment
#   Output:
#       - 3D reconstructed object
#
# ==============================================================================


# title: Load Packages and Define Functions
import os
import time
import tifffile as tiff

import torch
import numpy as np
import torch.nn as nn

from misc.models import *
from misc.utils import *
from misc.losses import *

data_fidelity_fun = {"mse": mse_loss, "mae": mae_loss, "ssim": ssim_loss}
import misc.visualize as vis
import misc.loading_data as ld
from misc.block_utils import Block_Scheduler
from misc.model_namer import pretrain_model_path, trained_model_path
from misc.propagator import PhysicalPropagator as PP
from misc.physics_params import PhysicsParamsManager

from tqdm import tqdm
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
)

dtype = torch.cuda.FloatTensor
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

torch.manual_seed(77)
np.random.seed(77)


class Logger:
    # class for logging and visualization
    _instance = None

    @staticmethod
    def get_instance():
        if Logger._instance is None:
            Logger()
        return Logger._instance

    def __init__(self):
        if Logger._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Logger._instance = self
        self.enabled = True

    def set_ability(self, enable=True):
        self.enabled = enable

    def dispaly_MIP(self, obj, title, mode, cmap, save_dir):
        if not self.enabled:
            return
        vis.display_stack_MIP(
            obj=obj, title=title, mode=mode, cmap=cmap, save_dir=save_dir
        )

    def display_multiMIP(self, objs, title, mode, cmap, save_dir):
        if not self.enabled:
            return
        vis.display_multistack_MIP(
            objs=objs, title=title, mode=mode, cmap=cmap, save_dir=save_dir
        )

    def display_curve(self, data, title, save_dir, metric):
        if not self.enabled:
            return
        vis.display_curve(loss=data, title=title, save_dir=save_dir, metric=metric)

    def save_stack(self, obj, title, save_dir):
        if not self.enabled:
            return
        tiff.imsave(os.path.join(save_dir, title + ".tif"), obj)


class Micro3DReconstructor(nn.Module):
    """
    Class representing a micro 3D reconstructor.

    Args:
        args: An object containing the arguments for the reconstructor.

    Attributes:
        args: An object containing the arguments for the reconstructor.
        working_type: The working type.
        projection_type: The projection type.
        net_obj_save_path_pretrained: The path to save the pretrained network object.
        net_obj_save_path_trained: The path to save the trained network object.
        ref_exist: A boolean indicating if the reference exists.
    """

    def __init__(
        self, args, DataLoader: ld.MicroDataLoader, PhysManager: PhysicsParamsManager, device=DEVICE
    ):
        super().__init__()
        self.DataLoader = DataLoader
        self.args = args
        self.PhysManager = PhysManager
        self.working_type = args.working_type
        self.projection_type = args.projection_type
        self.net_obj_save_path_pretrained_prefix = (
            args.net_obj_save_path_pretrained_prefix
        )
        self.device = device
        
        if args.log_option == "False":
            Logger.get_instance().set_ability(False)
        else:
            Logger.get_instance().set_ability(True)

        def name2index(name):
            basename = name.split(".")[-2]
            i_index = basename.split("-")[-2]
            j_index = basename.split("-")[-1]
            return int(i_index), int(j_index)
        
        self.i_index, self.j_index = name2index(args.data_stack_name)

        print(
            "Soure data dir: {},\n Saving dir: {},\n Step image dir: {}".format(
                args.root_dir, args.save_dir, args.step_image_dir
            )
        )
        self.net_obj_save_path_pretrained = pretrain_model_path(args, "Seminerf")
        self.net_obj_save_path_trained = trained_model_path(args, "Seminerf")
        self.psf_shape = args.psf_shape

        print(
            "Pretrained network object save path: {}".format(
                self.net_obj_save_path_pretrained
            )
        )
        print(
            "Trained network object save path: {}".format(
                self.net_obj_save_path_trained
            )
        )

        data_pack = self.load_data()
        self.y_max = data_pack["y_max"]
        self.y_min = data_pack["y_min"]
        self.init_max = data_pack["init_max"]

        # load coord
        axial_padding = args.axial_pad_length
        lateral_padding = args.lateral_pad_length
        INPUT_DEPTH, INPUT_WIDTH, INPUT_HEIGHT = data_pack["y_shape"]
        self.res = (args.psf_dz, args.psf_dx, args.psf_dy)
        self.padding_array = (axial_padding, lateral_padding, lateral_padding)
        self.regularize_array = (axial_padding-2, lateral_padding-2, lateral_padding-2)
        self.recover_shape = (INPUT_DEPTH, INPUT_WIDTH, INPUT_HEIGHT)
        self.coord_shape = (
            INPUT_DEPTH + axial_padding,
            INPUT_WIDTH + lateral_padding,
            INPUT_HEIGHT + lateral_padding,
        )
        print("Coordinates shape: {}".format(self.coord_shape))
        print("Effective recovery shape: {}".format(self.recover_shape))

        space_region = input_space_3d(*self.coord_shape)  # [d, h, w, 3]
        self.coordinates = space2coord(space_region).to(self.device)  # [d*h*w, 3]

        # Load model
        freq_logscale = True if args.freq_logscale == "True" else False
        self.net = Seminerf(
            D=args.nerf_num_layers,
            W=args.nerf_num_filters,
            skips=args.nerf_skips,
            out_channels=1,
            encoding_mode=args.encoding_option, # 
            radial_encoding_angle=args.radial_encoding_angle,
            radial_encoding_depth=args.radial_encoding_depth,
            cartesian_encoding_depth=args.cartesian_encoding_depth,
            zenith_encoding_angle=args.zenith_encoding_angle,
            gaussian_scale=args.gaussian_scale,
            gaussian_num=args.gaussian_encoding_num,
            freq_logscale=freq_logscale, 
            phys_params=self.PhysManager.ENC_params(self.coord_shape),
            device=self.device,
        ).to(self.device)

        torch.cuda.empty_cache()  # Release useless GPU memory

    def load_data(self):
        """
        Load the data for the reconstructor.

        Returns:
            data_pack: A dictionary containing the loaded data.
        """
        data_pack = self.DataLoader.load(
            PSF_shape=self.psf_shape,
        )
        psf_ = data_pack["psf"]
        y_ = data_pack["y"]
        rl_ = data_pack["rl"]
        init_data_ = data_pack["init_data"]
        print("Block size: {}, PSF size: {}".format(y_.shape, psf_.shape))

        self.ref_exist = False
        if "ref" in data_pack.keys():
            ref_ = data_pack["ref"]
            self.ref_exist = True
        elif os.path.exists(self.args.ref_path):
            ref_ = tiff.imread(self.args.ref_path).astype(np.float32)
            ref_ = ref_ / ref_.max()
            data_pack["ref"] = ref_
            self.ref_exist = True
        else:
            ref_ = None
        self.ref = ref_

        epsilon = 1e-3
        saving_dic = {
            "obj": np.log(psf_ + epsilon),
            "mode": "max",
            "cmap": "gray",
            "title": "PSF",
            "save_dir": self.args.save_dir,
        }
        vis.display_stack_MIP(**saving_dic)
        saving_dic["obj"] = np.log(np.abs(np.fft.fftshift(np.fft.fftn(psf_))) + epsilon)
        saving_dic["title"] = "OTF"
        vis.display_stack_MIP(**saving_dic)
        saving_dic["obj"] = y_
        saving_dic["title"] = "Measurement"
        saving_dic["mode"] = self.projection_type
        vis.display_stack_MIP(**saving_dic)

        self.psf = torch.from_numpy(psf_).type(dtype).to(self.device)
        self.y = torch.from_numpy(y_).type(dtype).to(self.device)
        self.rl = (
            torch.from_numpy(rl_).type(dtype).to(self.device) if rl_ is not None else None
        )  # [z, y, x]
        self.init_data = torch.from_numpy(init_data_).type(dtype).to(self.device)

        if self.ref_exist:
            self.ref = torch.from_numpy(ref_).type(dtype).to(self.device)
        else:
            self.ref = None

        return data_pack

    def infer(self, coord):
        out_x = self.net(coord)
        if self.args.nerf_beta is None:
            out_x = self.args.nerf_max_val * nn.Sigmoid()(out_x)
        else:
            out_x = nn.Softplus(beta=self.args.nerf_beta)(out_x)
            out_x = torch.minimum(torch.full_like(out_x, self.args.nerf_max_val), out_x)

        out_x = out_x.view(*self.coord_shape)

        return out_x

    def set_optim(self, state="train"):
        args = self.args
        if state == "pretrain":
            lr = args.pretraining_lr
            num_iter = args.pretraining_num_iter
        elif state == "train":
            lr = args.training_lr_obj
            num_iter = args.training_num_iter
        else:
            raise ValueError("Invalid state in func `Micro3DReconstructor.set_optim`")

        optimizer = torch.optim.Adam(
            [{"params": self.net.parameters(), "lr": lr}],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        if args.lr_schedule == "cosine":
            scheduler = CosineAnnealingLR(optimizer, num_iter, lr / 25)
        elif args.lr_schedule == "multi_step":
            scheduler = MultiStepLR(
                optimizer,
                milestones=[num_iter // 2, num_iter * 3 // 4],
                gamma=0.1,
            )

        return optimizer, scheduler

    def pretrain(self):
        """
        Pretrain the network to fit the measurement.
        Just fitting, no physical propagation after network.

        Args:
            y_max: The maximum value of the initial guess.
        """
        args = self.args
        optimizer, scheduler = self.set_optim(state="pretrain")

        loss_list = np.empty(shape=(1 + args.pretraining_num_iter,))
        loss_list[:] = np.NaN

        tbar = tqdm(
            range(args.pretraining_num_iter + 1),
        )

        win_params = self.PhysManager.WIN_params(self.recover_shape)
        fdmae_loss = FDMAE_Loss(self.recover_shape, win_params, device=self.device)
        hybrid_loss = lambda x, y: (ssim_loss(x,y) + fdmae_loss(x,y)) / 2
        pretrain_loss = {"hybrid": hybrid_loss, "mse": mse_loss, "ssim_loss": ssim_loss}[args.pretrain_loss]
        # init_data = fill_stack(self.init_data, self.padding_array)
        init_data = self.init_data
        print("Debugging: ", init_data.max(), self.init_max, self.y_min, self.y_max)
        for step in tbar:
            out_x = self.infer(self.coordinates)

            out_x_m = crop_stack(out_x, self.padding_array)
            # out_x_m = out_x
            loss = pretrain_loss(out_x_m , init_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_list[step] = loss.item()
            out_x_max = out_x_m.max().detach().cpu().numpy()
            out_x_min = out_x_m.min().detach().cpu().numpy()

            tbar.set_description(
                "Loss: %.6f|x_max: %.6f|x_min: %.6f"
                % (loss.item(), out_x_max, out_x_min)
            )
            tbar.refresh()

        fit_y_np = out_x_m.detach().cpu().numpy()
        vis.display_stack_MIP(
            fit_y_np, "Pretrain Fit", self.projection_type, "gray", args.save_dir
        )
        vis.display_curve(
            -10 * np.log10(loss_list), "-log10 of Pretrain Loss", args.save_dir, "loss"
        )

        return None

    def train(self):
        """
        Formal training process. No explicit return value, outputs are saved in the class.

        Outputs:
            self.recover_obj: The recovered object.
            self.pred_meas: The predicted measurement.
            self.full_obj: The full recovered area, bigger than object.
            self.full_pred_meas: The full predicted propagation measurement.
        """
        # Formal training process
        args = self.args
        optimizer, scheduler = self.set_optim(state="train")

        loss_list = np.empty(shape=(1 + args.training_num_iter,))
        loss_list[:] = np.NaN
        loss_val_list = np.empty(shape=(1 + args.training_num_iter,))
        loss_val_list[:] = np.NaN

        propagator = PP(self.psf, psf_shape=self.psf_shape, obj_shape=self.coord_shape)
        win_params = self.PhysManager.WIN_params(self.recover_shape)
        fdmae_loss = FDMAE_Loss(self.recover_shape, win_params, device=self.device)

        self.net.train()
        tbar = tqdm(
            range(args.training_num_iter + 1),
        )
        for step in tbar:
            obj = self.infer(self.coordinates)

            meas_pred = propagator.propagate(obj)
            useful_meas_pred = crop_stack(meas_pred, self.padding_array)
            useful_obj = crop_stack(obj, self.padding_array)

            pred = (useful_meas_pred + self.y_min) / self.y_max
            data_loss = (
                data_fidelity_fun[args.data_fidelity_term](pred, self.y)
                + fdmae_loss(pred, self.y) * args.fdmae_loss_weight
            )
            # reg_obj = crop_stack(obj, self.regularize_array)
            reg_obj = obj
            reg_loss = (
                hessian_loss(reg_obj, args.hessian_z_scale) * args.hessian_weight
                + mae_loss(reg_obj, 0) * args.l1_weight
            )
            loss = data_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if step in args.save_step_list:
                    Logger.get_instance().dispaly_MIP(
                        useful_obj.detach().cpu().numpy(),
                        "network_output,step:%d" % step,
                        self.projection_type,
                        "gray",
                        args.step_image_dir,
                    )
                    Logger.get_instance().dispaly_MIP(
                        useful_meas_pred.detach().cpu().numpy(),
                        "conv_output,step:%d" % step,
                        self.projection_type,
                        "gray",
                        args.step_image_dir,
                    )
                loss_list[step] = data_loss.item()
                obj_max = torch.max(useful_obj).item()
                obj_min = torch.min(useful_obj).item()
                pred_max = torch.max(useful_meas_pred).item()
                pred_min = torch.min(useful_meas_pred).item()
                if self.ref_exist:
                    val_loss = torch.mean(
                        torch.square(useful_obj / useful_obj.max() - self.ref)
                    ).item()
                    loss_val_list[step] = val_loss
                    tbar_str = (
                        "loss=%.3e|data=%.3e|max_x=%.3e|min_x=%.3e|val_psnr=%.2fdB"
                        % (
                            loss.item(),
                            data_loss.item(),
                            obj_max,
                            obj_min,
                            -10 * np.log10(val_loss),
                        )
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

        vis.display_curve(
            -10 * np.log10(loss_list), "Training Loss Log10", args.save_dir, "loss"
        )
        if self.ref_exist:
            vis.display_curve(
                -10 * np.log10(loss_val_list), "Validation PSNR", args.save_dir, "PSNR"
            )
            val_loss = torch.mean(
                torch.square(useful_obj / useful_obj.max() - self.ref)
            ).item()
            print("Validation PSNR: %.3fdB" % (-10 * np.log10(val_loss)))
        else:
            print("No reference data, skip validation.")
            loss = torch.mean(
                torch.square((useful_meas_pred + self.y_min) / self.y_max - self.y)
            ).item()
            print("PSNR comparing with measurement: %.3fdB" % (-10 * np.log10(loss)))

        print("Max value of reconstructed object: %.3e" % torch.max(useful_obj).item())

        print("Training finished.")
        self.recover_obj = useful_obj.detach().cpu().numpy() + self.y_min
        self.pred_meas = useful_meas_pred.detach().cpu().numpy()
        self.full_obj = obj.detach().cpu().numpy()
        self.full_pred_meas = meas_pred.detach().cpu().numpy()

        return None

    def post_process(self):
        # Post process: visualization, save results, etc.
        args = self.args
        y = self.y.detach().cpu().numpy()
        # Visualization: MIP along z, y, x-axes. Comparing the input image stack and the reconstructed structure.
        if self.ref_exist:
            ref = self.ref.detach().cpu().numpy()
            vis.display_multistack_MIP(
                [y, self.pred_meas, self.recover_obj, ref],
                "meas-pred_meas-obj-ref",
                self.projection_type,
                "gray",
                args.save_dir,
            )
            # print the PSNR between the recovered object and the reference object
            psnr = -10 * np.log10(
                np.mean(np.square(self.recover_obj / self.recover_obj.max() - ref))
            )
            print("Validation PSNR: %.3fdB on block (%d, %d)" % (psnr, self.i_index, self.j_index))
        else:
            vis.display_multistack_MIP(
                [y, self.pred_meas, self.recover_obj],
                "meas-pred_meas-obj",
                self.projection_type,
                "viridis",
                args.save_dir,
            )
        save_rec = self.recover_obj
        Logger.get_instance().save_stack(
            save_rec, "rec_" + args.data_stack_name.split(".")[0], args.save_dir
        )
        return None

    def forward(self):
        # if not need pretrain
        if not self.args.pretraining:
            pass
        # if need pretrain, need load previously pretrained model, and there is already a previously pretrained network
        elif self.args.loading_pretrained_model == "True" and os.path.exists(
            self.net_obj_save_path_pretrained
        ):
            self.net.load_state_dict(torch.load(self.net_obj_save_path_pretrained))
            print("Pre-trained model loaded.")
        # if need pretrain, but not need previously pretrained model, or there is no previously pretrained network
        else:
            self.pretrain()
            if self.args.saving_model == "True":
                torch.save(self.net.state_dict(), self.net_obj_save_path_pretrained)
                print(
                    "Pretrained network saved to " + self.net_obj_save_path_pretrained
                )
            else:
                print("Pretrained network not saved.")

        # Formal training process
        self.train()
        if self.args.saving_model == "True":
            torch.save(self.net.state_dict(), self.net_obj_save_path_trained)
            print("Trained network saved to " + self.net_obj_save_path_trained)
        else:
            print("Trained network not saved.")

        self.post_process()

        return self.recover_obj, self.i_index, self.j_index


def reconstruct_whole_FOV(args, i_index: list = [], j_index: list = []):
    """
    Receive a large FOV measurements, reconstruct it split and merge strategy.
    """
    pretraining_iter = args.pretraining_num_iter
    training_iter = args.training_num_iter
    args.save_dir = os.path.join(args.exp_dir, args.exp_name)
    args.step_image_dir = os.path.join(args.save_dir, "step_images")
    physics_manager = PhysicsParamsManager(args)

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.step_image_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)

    # title: Read in whole FOV measurement
    whole_FOV = tiff.imread(os.path.join(args.root_dir, args.data_stack_name))
    whole_FOV = whole_FOV.astype(np.float32) / whole_FOV.max()
    block_spliter = Block_Scheduler(
        whole_FOV,
        args.axial_view,
        args.lateral_view,
        args.lateral_overlap,
        mask_mode=args.mask_mode,
    )

    if (
        os.path.exists(os.path.join(args.root_dir, args.init_stack_name))
        and args.init_stack_name != args.data_stack_name
    ):
        init_FOV = tiff.imread(os.path.join(args.root_dir, args.init_stack_name))
        init_FOV = init_FOV.astype(np.float32) / init_FOV.max()
        initial_spliter = Block_Scheduler(
            init_FOV,
            args.axial_view,
            args.lateral_view,
            args.lateral_overlap,
            mask_mode=args.mask_mode,
        )
    else:
        init_FOV = whole_FOV.copy()
        initial_spliter = block_spliter

    if os.path.exists(os.path.join(args.root_dir, args.ref_name)):
        whole_ref = tiff.imread(os.path.join(args.root_dir, args.ref_name))
        whole_ref = whole_ref.astype(np.float32) / whole_ref.max()
        reference_spliter = Block_Scheduler(
            whole_ref,
            args.axial_view,
            args.lateral_view,
            args.lateral_overlap,
            mask_mode=args.mask_mode,
        )
    else:
        reference_spliter = lambda i, j: None

    if os.path.exists(os.path.join(args.root_dir, args.rl_stack_name)):
        whole_rl = tiff.imread(os.path.join(args.root_dir, args.rl_stack_name))
        whole_rl = whole_rl.astype(np.float32) / whole_rl.max()
        rl_spliter = Block_Scheduler(
            whole_rl,
            args.axial_view,
            args.lateral_view,
            args.lateral_overlap,
            mask_mode=args.mask_mode,
        )
    else:
        rl_spliter = lambda i, j: None

    row_num = block_spliter.height_block_num
    col_num = block_spliter.width_block_num

    i_index_iterable = range(row_num) if len(i_index) == 0 else i_index
    j_index_iterable = range(col_num) if len(j_index) == 0 else j_index

    start_time = time.time()
    last_time = start_time

    for i in i_index_iterable:  # height
        for j in j_index_iterable:  # width
            # title: Crop the sub-measurement
            sub_block = block_spliter(i, j)
            # sub_block_init = sub_block.copy()
            sub_block_init = initial_spliter(i, j)
            reference_block = reference_spliter(i, j)
            rl_block = rl_spliter(i, j)
            sub_block_var = sub_block.var()
            sub_block_mean = sub_block.mean()
            args.data_stack_name = args.data_stack_name.split(".")[0].split("-")[
                0
            ] + "-{}-{}.tif".format(i, j)
            print(
                "Variance of block: {}; Average of block: {}".format(
                    sub_block.var(), sub_block.mean()
                )
            )
            if (
                sub_block_var < args.pure_background_variance_gate
                and sub_block_mean < args.pure_background_mean_gate
            ):
                args.pretraining_num_iter = int(pretraining_iter / 10)
                args.training_num_iter = int(training_iter / 10)
                args.pretrain_loss = "mse"
            else:
                args.pretraining_num_iter = pretraining_iter
                args.training_num_iter = training_iter
                args.pretrain_loss = "hybrid"

            # title: Reconstruct the sub-measurement
            loader = ld.MicroDataLoader(
                args.working_type,
                args,
                physics_manager.PSF_params(),
                measurement=sub_block,
                init_guess=sub_block_init,
                ref=reference_block,
                rl=rl_block,
            )
            ReConstructor = Micro3DReconstructor(args, loader, physics_manager)
            print(f"***Start processing {args.data_stack_name}, h_idx:{i}, w_idx:{j}.")
            recover_block, _, _ = ReConstructor()
            print(f"***Finish processing {args.data_stack_name}, h_idx:{i}, w_idx:{j}.")

            block_spliter.feedback_recovered_block(i, j, recover_block)
            whole_FOV_rec = block_spliter.get_reconstructed_stack()
            vis.display_stack_MIP(
                whole_FOV_rec, "Whole FOV Reconstruction", "max", "gray", args.save_dir
            )
            whole_FOV_rec_8b = whole_FOV_rec / whole_FOV_rec.max() * 255
            whole_FOV_rec_8b = whole_FOV_rec_8b.astype(np.uint8)
            tiff.imsave(
                os.path.join(args.save_dir, "rec_whole_FOV_8b.tif"),
                whole_FOV_rec_8b,
            )

            temp_time = time.time()
            print(f"Time for h{i}-w{j}: {(temp_time - last_time) / 60:.2f}min")
            last_time = temp_time
            print("----------------------------------------------")

    end_time = time.time()
    print(f"Total time consuming: {(end_time - start_time) / 60:.2f}min")

    print("MAX of whole FOV block: {}".format(whole_FOV_rec.max()))
    tiff.imsave(
        os.path.join(args.save_dir, "rec_whole_FOV.tif"),
        whole_FOV_rec.astype(np.float32),
    )
    whole_FOV_freq = np.fft.fftshift(np.fft.fftn(whole_FOV_rec))
    vis.display_stack_MIP(np.log(np.abs(whole_FOV_freq)), "Whole FOV Frequency", "max", "viridis", args.save_dir, axial_sclae_ratio=4)
    print("Reconstruction of whole FOV measurement FINISHED.")

    return None


if __name__ == "__main__":
    # title: Set Arguments
    import warnings
    import time

    warnings.filterwarnings("ignore")

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    from opt import init_opts, load_opts

    args = init_opts()

    args.psf_path = os.path.join(args.root_dir, args.psf_name)
    args.ref_path = os.path.join(args.root_dir, args.ref_name)
    args.pretrain_loss = "mse"
    if not os.path.exists(args.net_obj_save_path_pretrained_prefix):
        os.makedirs(args.net_obj_save_path_pretrained_prefix)

    args.save_step_list = [100 * i for i in range(6)] + [500 * i for i in range(2, 20)]

    if args.log_option == "False":
        Logger.get_instance().set_ability(False)
    else:
        Logger.get_instance().set_ability(True)

    reconstruct_whole_FOV(args, i_index=args.row_picker, j_index=args.col_picker)

