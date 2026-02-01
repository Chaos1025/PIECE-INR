"""
ProgressManager: 统一的进度条管理器
用于多进程图像块处理时显示进度信息
"""

from tqdm import tqdm
from typing import Dict, Optional
import threading


class ProgressManager:
    """统一的进度条管理器 (主进程端)

    管理所有 GPU 的进度条显示：
    - 每个 GPU 一个块进度条（累积显示处理进度）
    - 每个 GPU 一个块内迭代进度条（随块更新而刷新）
    """

    def __init__(self):
        self.gpu_pbars: Dict[int, tqdm] = {}  # GPU 块进度条
        self.block_pbars: Dict[int, tqdm] = {}  # GPU 当前块迭代进度条 (每个GPU一行)
        self.block_counts: Dict[int, int] = {}  # 每个GPU处理的块数
        self.total_blocks: Dict[int, int] = {}  # 每个GPU总块数
        # 跟踪当前正在处理的块和阶段，用于验证进度更新
        self.current_block: Dict[int, tuple] = {}  # gpu_id -> (block_id, phase)
        self._lock = threading.Lock()

    def create_gpu_progress(self, gpu_id: int, total_blocks: int):
        """为 GPU 创建块进度条和块内迭代进度条

        Args:
            gpu_id: GPU ID
            total_blocks: 该 GPU 需要处理的块数
        """
        with self._lock:
            self.block_counts[gpu_id] = 0
            self.total_blocks[gpu_id] = total_blocks
            self.current_block[gpu_id] = (None, None)

            # 块进度条 (position=0, 2, 4...)
            self.gpu_pbars[gpu_id] = tqdm(
                total=total_blocks,
                desc=f"GPU {gpu_id}",
                position=gpu_id * 2,
                leave=True,
                colour="cyan",
                ascii=True,
            )

            # 块内迭代进度条 (position=1, 3, 5...) - 初始不显示
            self.block_pbars[gpu_id] = tqdm(desc="Block -", position=gpu_id * 2 + 1, leave=False)

    def start_block(self, gpu_id: int, block_id: int, total_iter: int, phase: str):
        """开始处理新块 (pretrain 或 train 阶段)

        Args:
            gpu_id: GPU ID
            block_id: 块 ID
            total_iter: 该阶段的总迭代次数
            phase: 阶段名称 ('pretrain' 或 'train')
        """
        with self._lock:
            self.block_counts[gpu_id] += 1
            self.gpu_pbars[gpu_id].update(1)

            # 更新当前处理的块和阶段
            self.current_block[gpu_id] = (block_id, phase)

            pbar = self.block_pbars[gpu_id]
            pbar.reset(total=total_iter)
            pbar.n = 0  # 重置进度
            pbar.set_description(f"Block {block_id} ({phase})")

    def switch_phase(self, gpu_id: int, block_id: int, total_iter: int, new_phase: str):
        """切换阶段 (pretrain -> train)

        Args:
            gpu_id: GPU ID
            block_id: 块 ID
            total_iter: 新阶段的总迭代次数
            new_phase: 新阶段名称 ('train')
        """
        with self._lock:
            # 更新当前阶段
            old_block_id, _ = self.current_block.get(gpu_id, (block_id, None))
            self.current_block[gpu_id] = (old_block_id, new_phase)

            pbar = self.block_pbars[gpu_id]
            pbar.reset(total=total_iter)
            pbar.n = 0  # 重置进度
            pbar.set_description(f"Block {block_id} ({new_phase})")

    def update_iter_progress(
        self, gpu_id: int, block_id: int, phase: str, step: int, total: int, desc: str = ""
    ):
        """更新当前块的迭代进度（带验证，确保不冲突）

        Args:
            gpu_id: GPU ID
            block_id: 块 ID
            phase: 阶段名称 ('pretrain' 或 'train')
            step: 当前迭代步数
            total: 总迭代次数
            desc: 描述信息 (如 loss 值)
        """
        with self._lock:
            # 验证这个更新是否匹配当前正在处理的块和阶段
            current = self.current_block.get(gpu_id)
            if current is None:
                return

            current_block_id, current_phase = current
            # 如果不匹配，跳过这个更新（可能是一个延迟的旧进度更新）
            if current_block_id != block_id or current_phase != phase:
                return

            if gpu_id in self.block_pbars:
                pbar = self.block_pbars[gpu_id]
                pbar.n = step
                pbar.total = total  # 确保 total 正确
                pbar.refresh()
                if desc:
                    pbar.set_description(f"Block {block_id} ({phase}): {desc}")

    def close(self):
        """关闭所有进度条"""
        for pbar in self.block_pbars.values():
            try:
                pbar.close()
            except Exception:
                pass
        for pbar in self.gpu_pbars.values():
            try:
                pbar.close()
            except Exception:
                pass
