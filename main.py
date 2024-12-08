""" 
Main program for reconstruction
Both single and multi-process reconstruction are supported.
"""

import os
import multiprocessing as mp
import time
import warnings
import random
from argparse import Namespace
import copy
import tifffile as tiff

from misc.utils import *
import misc.visualize as vis
import misc.loading_data as ld
from misc.block_utils import Block_Scheduler
from misc.metrics import reconstruction_metric
from misc.physics_params import PhysicsParamsManager

from opt import init_opts, load_opts
from fluor_rec3d import Micro3DReconstructor, Logger

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(77)


def index2ij(idx, row_num):
    return idx // row_num, idx % row_num


def ij2index(i, j, row_num):
    return i * row_num + j


def null_func(*args, **kwargs):
    pass


def process_image_block(
    block_list,
    gpu_id,
    block_splitter,
    initial_splitter,
    refrence_splitter,
    args,
    physics_manager,
):

    pretraining_num_iter = args.pretraining_num_iter
    training_num_iter = args.training_num_iter

    row_num = block_splitter.height_block_num
    col_num = block_splitter.width_block_num

    device = torch.device(f"cuda:{gpu_id}")
    print(f"WORKING GPU ID: {gpu_id}")
    rec_blocks = {}
    for block_id in block_list:
        i, j = index2ij(block_id, row_num)
        block = block_splitter(i, j)
        init_block = initial_splitter(i, j)
        ref_block = refrence_splitter(i, j)
        block_var = block.var()
        block_mean = block.mean()
        args.data_stack_name = args.data_stack_name.split(".")[0].split("-")[
            0
        ] + "-{}-{}.tif".format(i, j)

        if (
            block_var < args.pure_background_variance_gate
            and block_mean < args.pure_background_mean_gate
        ):
            args.pretraining_num_iter = int(pretraining_num_iter / 10)
            args.training_num_iter = int(training_num_iter / 10)
            args.pretrain_loss = "mse"
        else:
            args.pretraining_num_iter = pretraining_num_iter
            args.training_num_iter = training_num_iter
            args.pretrain_loss = "hybrid"

        loader = ld.MicroDataLoader(
            args.working_type,
            args,
            physics_manager.PSF_params(),
            measurement=block,
            init_guess=init_block,
            ref=ref_block,
        )

        print(f"Processing image block {block_id} on GPU {gpu_id}")

        ReConstructor = Micro3DReconstructor(args, loader, physics_manager, device)
        rec_block, i, j = ReConstructor()
        rec_blocks[block_id] = rec_block

        block_splitter.feedback_recovered_block(i, j, rec_block)
        merge_block = block_splitter.get_reconstructed_stack()
        vis.display_stack_MIP(
            merge_block, f"Whole_FOV-merge-GPU{gpu_id}", "max", "gray", args.save_dir
        )

    return rec_blocks


def main(args, i_index: list = [], j_index: list = [], gpu_list: list = []):
    """
    Receive a large FOV measurements, reconstruct it split and merge strategy.
    """
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
    block_splitter = Block_Scheduler(
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
        initial_splitter = Block_Scheduler(
            init_FOV,
            args.axial_view,
            args.lateral_view,
            args.lateral_overlap,
            mask_mode=args.mask_mode,
        )
    else:
        init_FOV = whole_FOV.copy()
        initial_splitter = block_splitter

    if os.path.exists(os.path.join(args.root_dir, args.ref_name)):
        whole_ref = tiff.imread(os.path.join(args.root_dir, args.ref_name))
        whole_ref = whole_ref.astype(np.float32) / whole_ref.max()
        reference_splitter = Block_Scheduler(
            whole_ref,
            args.axial_view,
            args.lateral_view,
            args.lateral_overlap,
            mask_mode=args.mask_mode,
        )
        metric_device = (
            torch.device("cuda:{}".format(gpu_list[0]))
            if len(gpu_list) > 0
            else torch.device("cpu")
        )
        recon_metric = reconstruction_metric(metric_device)
    else:
        reference_splitter = null_func

    if os.path.exists(os.path.join(args.root_dir, args.rl_stack_name)):
        whole_rl = tiff.imread(os.path.join(args.root_dir, args.rl_stack_name))
        whole_rl = whole_rl.astype(np.float32) / whole_rl.max()
        rl_splitter = Block_Scheduler(
            whole_rl,
            args.axial_view,
            args.lateral_view,
            args.lateral_overlap,
            mask_mode=args.mask_mode,
        )
    else:
        rl_splitter = null_func

    row_num = block_splitter.height_block_num
    col_num = block_splitter.width_block_num

    row_iter_num = len(i_index) if len(i_index) != 0 else row_num
    col_iter_num = len(j_index) if len(j_index) != 0 else col_num

    i_index_iterable = range(row_num) if len(i_index) == 0 else i_index
    j_index_iterable = range(col_num) if len(j_index) == 0 else j_index

    start_time = time.time()

    block_args = Namespace(**copy.deepcopy(vars(args)))

    global_block_list = []  # global block index need to be processed
    for i in i_index_iterable:
        for j in j_index_iterable:
            global_block_list.append(ij2index(i, j, row_num))

    blocks_process_indicator = np.zeros((row_num, col_num), dtype=np.int8) - 1
    for block_id in global_block_list:
        i, j = index2ij(block_id, row_num)
        block = block_splitter(i, j)
        block_var = block.var()
        block_mean = block.mean()
        if (
            block_var < args.pure_background_variance_gate
            and block_mean < args.pure_background_mean_gate
        ):  # pure background block
            blocks_process_indicator[i, j] = 0
        else:
            blocks_process_indicator[i, j] = 1
    np.savetxt(
        os.path.join(args.save_dir, "blocks_process_indicator.txt"),
        np.c_[blocks_process_indicator],
        fmt="%d",
        delimiter="\t",
    )

    # title: Multi-processing initialization
    num_gpus = len(gpu_list)
    num_blocks = row_iter_num * col_iter_num
    # randomly shuffle the block list, for better load balance
    random.shuffle(global_block_list)

    blocks_per_gpu = num_blocks // num_gpus  # ignore the remainder temporarily
    pool = mp.Pool(processes=num_gpus)
    results = []
    # distribute the blocks to GPUs
    gpu_block_dict = {}
    for gpu_idx, gpu_id in enumerate(gpu_list):
        start_block = gpu_idx * blocks_per_gpu
        end_block = start_block + blocks_per_gpu
        gpu_block_dict[gpu_id] = [
            global_block_list[idx] for idx in range(start_block, end_block)
        ]
    # remaining blocks for the last GPU
    remaining_blocks = num_blocks % num_gpus
    if remaining_blocks > 0:
        for block_id in range(num_blocks - remaining_blocks, num_blocks):
            gpu_block_dict[gpu_list[-1]].append(global_block_list[block_id])

    # title: Multi-processing
    for gpu_id in gpu_list:
        current_block_list = gpu_block_dict[gpu_id]

        results.append(
            pool.apply_async(
                process_image_block,
                (
                    current_block_list,
                    gpu_id,
                    block_splitter,
                    initial_splitter,
                    reference_splitter,
                    block_args,
                    physics_manager,
                ),
            )
        )

    pool.close()
    pool.join()

    # title: Merge blocks and post processing
    for r in results:
        blocks = r.get()
        for block_id, block in blocks.items():
            i, j = index2ij(block_id, row_num)
            block_splitter.feedback_recovered_block(i, j, block)

    whole_FOV_rec = block_splitter.get_reconstructed_stack()
    vis.display_stack_MIP(
        whole_FOV_rec, "Whole FOV Reconstruction", "max", "gray", block_args.save_dir
    )
    if os.path.exists(os.path.join(args.root_dir, args.ref_name)):
        recon_metric.get_metric(
            whole_ref, whole_FOV_rec / whole_FOV_rec.max(), "Whole FOV"
        )

    whole_FOV_rec_8b = whole_FOV_rec / whole_FOV_rec.max() * 255
    whole_FOV_rec_8b = whole_FOV_rec_8b.astype(np.uint8)
    tiff.imsave(
        os.path.join(block_args.save_dir, "rec_whole_FOV_8b.tif"),
        whole_FOV_rec_8b,
    )
    print("Max value of whole FOV reconstruction: ", whole_FOV_rec.max())
    tiff.imsave(
        os.path.join(block_args.save_dir, "rec_whole_FOV.tif"),
        whole_FOV_rec.astype(np.float32),
    )

    print("Total time consuming: {:.2f}min".format((time.time() - start_time) / 60))

    return None


if __name__ == "__main__":
    mp.set_start_method("spawn")

    warnings.filterwarnings("ignore")

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    args = init_opts()
    usable_gpus = args.gpu_list
    print("Usable GPUs ID: ", usable_gpus)
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

    main(args, i_index=args.row_picker, j_index=args.col_picker, gpu_list=usable_gpus)
