import torch
import numpy as np
import torch.nn as nn
from typing import Union


class Block_Scheduler(nn.Module):
    """Scheduler for block-wise reconstruction"""

    TOP_EDGE = 1
    BOTTOM_EDGE = 2
    LEFT_EDGE = 4
    RIGHT_EDGE = 8

    def __init__(
        self,
        whole_stack: np.array,
        axial_view: int,
        lateral_view: int,
        lateral_overlap: int,
        mask_mode: str = "smooth",
    ):
        """
        whole_stack: whole measurement stack, corresponding to the area for reconstruction
        axial_view: axial view size for a single reconstruction block, in stack domain
        lateral_view: lateral view size for a single reconstruction block, in stack domain
        lateral_overlap: overlap size between adjacent blocks, in stack domain
        mask_mode: mask mode for the overlap region, "smooth" or "steep"

        """
        super(Block_Scheduler, self).__init__()
        d, h, w = whole_stack.shape
        self.stack = whole_stack.astype(np.float32) / whole_stack.max()
        self.mask_mode = mask_mode
        self.axial_view = whole_stack.shape[0] if axial_view > whole_stack.shape[0] else axial_view
        self.lateral_view = h if lateral_view > h else lateral_view
        self.lateral_overlap = lateral_overlap
        self.step_size = self.lateral_view - self.lateral_overlap
        self.width_block_num = (w - self.lateral_view) // self.step_size + 1
        self.height_block_num = (h - self.lateral_view) // self.step_size + 1
        # self.lateral_block_num = (h - self.lateral_view) // self.step_size + 1

        self.rec_stack = np.zeros_like(whole_stack, np.float32)

    def get_global_mask(self):
        """
        Return a mask for the whole stack, with the same size as the stack
        Only for the `steep` mode mask.
        """
        global_mask = np.ones_like(self.rec_stack, np.float32)
        for i in range(self.height_block_num - 1):  # height, row
            global_mask[
                :,
                (i + 1) * self.step_size : (i + 1) * self.step_size + self.lateral_overlap,
                :,
            ] *= (
                1 / 2
            )
        for j in range(self.width_block_num - 1):  # width, column
            global_mask[
                :,
                :,
                (j + 1) * self.step_size : (j + 1) * self.step_size + self.lateral_overlap,
            ] *= (
                1 / 2
            )
        return global_mask

    def check_position(self, row_index: int, col_index: int):
        """Get blocks position in the whole stack
        Return position code, 4 bits, from right to left, [top, bottom, left, right]
        """
        position_flag = 0
        if row_index == 0:
            position_flag += self.TOP_EDGE
        if row_index == self.height_block_num - 1:
            position_flag += self.BOTTOM_EDGE
        if col_index == 0:
            position_flag += self.LEFT_EDGE
        if col_index == self.width_block_num - 1:
            position_flag += self.RIGHT_EDGE
        return position_flag

    def get_single_mask(self, row_index: int, col_index: int):
        """Get mask for a single block
        For both `smooth` and `steep` mode
        """
        position_flag = self.check_position(row_index, col_index)
        position_str = "{:04b}".format(position_flag)
        pos = [bool(int(bit)) for bit in position_str]  # [right, left, bottom, top]
        mask = np.ones((self.lateral_view, self.lateral_view), np.float32)

        # Gradual transition in boundary regions
        if self.mask_mode == "smooth":
            if not pos[3]:  # not top edge
                mask[: self.lateral_overlap, :] *= np.linspace(0, 1, self.lateral_overlap)[:, None]
            if not pos[2]:  # not bottom edge
                mask[-self.lateral_overlap :, :] *= np.linspace(1, 0, self.lateral_overlap)[:, None]
            if not pos[1]:  # not left edge
                mask[:, : self.lateral_overlap] *= np.linspace(0, 1, self.lateral_overlap)[None, :]
            if not pos[0]:  # not right edge
                mask[:, -self.lateral_overlap :] *= np.linspace(1, 0, self.lateral_overlap)[None, :]
        elif self.mask_mode == "steep":
            if not pos[3]:
                mask[: self.lateral_overlap, :] *= 1 / 2
            if not pos[2]:
                mask[-self.lateral_overlap :, :] *= 1 / 2
            if not pos[1]:
                mask[:, : self.lateral_overlap] *= 1 / 2
            if not pos[0]:
                mask[:, -self.lateral_overlap :] *= 1 / 2
        return mask

    def forward(self, row_index: int, col_index: int):
        """Get a single block by its row and column index"""
        # pick up corresponding block
        row_start = row_index * self.step_size
        row_end = row_start + self.lateral_view
        col_start = col_index * self.step_size
        col_end = col_start + self.lateral_view
        current_block = self.stack[:, row_start:row_end, col_start:col_end]

        # return np.clip(current_block, 0, 1)
        return current_block

    def feedback_recovered_block(self, row_index: int, col_index: int, recovered_block: np.array):
        """Feedback the recovered block to the stored recovery"""
        row_start = row_index * self.step_size
        row_end = row_start + self.lateral_view
        col_start = col_index * self.step_size
        col_end = col_start + self.lateral_view
        # generate mask with smooth transition
        mask = self.get_single_mask(row_index, col_index)
        self.rec_stack[:, row_start:row_end, col_start:col_end] += (
            recovered_block * mask[None, :, :]
        )
        return None

    def clear_recovered_blocks(self):
        self.rec_stack = np.zeros_like(self.stack, np.float32)
        return None

    def get_reconstructed_stack(self):
        return self.rec_stack


class Block_wise_Maker:
    """Split the whole stack into blocks and apply a specific function to each block"""

    def __init__(
        self,
        stack: np.array,
        axial_view: int,
        lateral_view: int,
        lateral_overlap: int,
    ):
        super(Block_wise_Maker, self).__init__()
        d, h, w = stack.shape
        self.axial_view = axial_view
        self.lateral_view = lateral_view
        self.lateral_overlap = lateral_overlap
        self.step_size = self.lateral_view - self.lateral_overlap
        self.width_block_num = (w - self.lateral_view) // self.step_size + 1
        self.height_block_num = (h - self.lateral_view) // self.step_size + 1
        self.stack = stack

    def make(self, func):
        ret_mat = np.zeros((self.height_block_num, self.width_block_num), np.float32)
        for i in range(self.height_block_num):  # row
            for j in range(self.width_block_num):  # column
                current_block = self.stack[
                    :,
                    i * self.step_size : i * self.step_size + self.lateral_view,
                    j * self.step_size : j * self.step_size + self.lateral_view,
                ]
                ret_mat[i, j] = func(current_block)
        return ret_mat

    def print(self, func):
        for i in range(self.height_block_num):  # row
            for j in range(self.width_block_num):  # column
                current_block = self.stack[
                    :,
                    i * self.step_size : i * self.step_size + self.lateral_view,
                    j * self.step_size : j * self.step_size + self.lateral_view,
                ]
                print(func(current_block))
        return None


class Global_Block_Scheduler:
    def __init__(
        self,
        stack: np.array,
        padding_array: Union[list, tuple],
        lateral_view: int,
        lateral_overlap: int,
    ):
        super(Global_Block_Scheduler, self).__init__()
        self.stack = stack
        self.padding_array = padding_array
        self.sample_range = stack.shape
        self.coord_range = [self.sample_range[i] + self.padding_array[i] for i in range(3)]
        self.lateral_view = lateral_view
        self.lateral_overlap = lateral_overlap
        self.step_size = self.lateral_view - self.lateral_overlap
        self.lateral_block_num = (stack.shape[1] - lateral_overlap) // self.step_size
