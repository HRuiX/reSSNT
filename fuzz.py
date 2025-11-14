import copy
import numpy as np
import time
import coverage
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from mmseg.apis import inference_model, init_model
import torch
import torch.nn as nn
import gc
from mmengine.registry import init_default_scope
import utility
import mmcv
from rich.console import Console

console = Console()

# Import logging system and display functions
from fuzzer_logger import (
    get_fuzzer_logger,
    display_coverage_summary,
    display_epoch_progress,
)

init_default_scope("mmseg")

import os
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class FuzzingConfig:
    alpha: float = 0.2
    beta: float = 0.4
    try_num: int = 50
    p_min: float = 0.01
    gamma: int = 5
    k: int = 64
    num_processes: int = 1
    batch_size: int = 1
    every_pic: int = 1
    max_epochs: int = 10000
    max_runtime_hours: int = 6


@dataclass
class EpochStats:
    epoch_num: int
    generated_count: int = 0
    valid_count: int = 0
    coverage_improvements: int = 0
    defect_detections: int = 0
    start_time: Optional[float] = None
    runtime: float = 0.0


def check_and_remove_file(path1):
    try:
        path2 = path1.replace("/leftImg8bit", "/gtFine")
        path2 = path2.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        if not os.path.exists(path1):
            return False

        if not os.path.exists(path2):
            os.remove(path1)
            return True
        else:
            return False

    except Exception as e:
        print(f"  ⚠️  处理文件时出错: {e}")
        return False


class PriorityScheduler:
    """优先级调度器"""

    def __init__(self, config: FuzzingConfig):
        self.config = config

    def calculate_priority(self, B_ci: float) -> float:
        """计算优先级"""
        threshold = (1 - self.config.p_min) * self.config.gamma
        if B_ci < threshold:
            return 1 - B_ci / self.config.gamma
        else:
            return self.config.p_min

    def select_next_batch(self, T: Tuple, batch_size: int) -> List:
        """选择下一批图像"""
        B_c, Bs = T
        B_p = [self.calculate_priority(B_c[i]) for i in range(len(B_c))]
        p_sum = np.sum(B_p)

        if p_sum == 0:
            # 如果所有权重都是0，使用均匀分布
            probs = None
        else:
            probs = B_p / p_sum

        idxs = np.random.choice(len(Bs), size=batch_size, p=probs, replace=False)
        return [Bs[idx] for idx in idxs]

    def power_schedule(self, S: List, K: int) -> callable:
        """计算功率调度"""
        potentials = []
        for I in S:
            I1, I0 = I.img, I.ori_img
            p = self.config.beta * 255 * np.sum(I1 > 0) - np.sum(np.abs(I1 - I0))
            potentials.append(p)

        potentials = np.array(potentials)
        potential_sum = np.sum(potentials)

        if potential_sum > 0:
            potentials = potentials / potential_sum
        else:
            potentials = np.ones_like(potentials) / len(potentials)

        def Ps(I_id: int) -> int:
            return int(np.ceil(potentials[I_id] * K))

        return Ps


class Fuzzer:
    """重构后的模糊测试器主类"""

    def __init__(self, model_name, config, checkpoint, num_classes, cov, dataset,model_type, save_data_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = FuzzingConfig()
        self.model = init_model(config, checkpoint, device=self.device)
        self.model.eval()
        self.num_classes = num_classes
        self.model_name = model_name
        self.dataset = dataset
        self.cov_type = f"{cov[0]}-{cov[1]}"

        # 设置路径
        self.fuzz_save_path_prefix = f"./fuzz-output-data-{save_data_path}/{self.cov_type}/{self.dataset}/{model_type}/"
        self.coverage_save_path = f"{self.fuzz_save_path_prefix}/{self.cov_type}/{dataset}"
        self.TOOL_LOG_FILE_PATH = f"./New_temp/{self.dataset}/{model_type}/{self.model_name}"

        # 创建必要的目录
        Path(self.coverage_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.TOOL_LOG_FILE_PATH).mkdir(parents=True, exist_ok=True)

        # 添加钩子
        self.output = None
        self.layer_output_input_dict = {}
        self.handles = []
        self.layer_dict = utility.get_model_layers(self.model )
        self.layer_dict_keys = list(self.layer_dict.keys())
        self.output_sizes = {}
        self.layer_dict_keys_cnt = 0
        self._get_output_size()
        self.layer_dict_keys_cnt = 0
        self.handles = []
        self.register_hooks()

        # 初始化 coverage criterion
        self.criterion = getattr(coverage, cov[0])(model_name=model_name, layer_size_dict=self.output_sizes,
                                                   device=self.device, save_path=self.coverage_save_path,
                                                   TOOL_LOG_FILE_PATH=self.TOOL_LOG_FILE_PATH, threshold=cov[1])

        # 初始化组件
        self.logger = self._initialize_logger()
        self.priority_scheduler = PriorityScheduler(self.config)

        # 状态变量
        self.epoch = 0
        self.delta_time = 0
        self.old_cov_record = None
        self.current_epoch_stats = None

        self.logger.log_message("Fuzzer initialized")

    def _get_output_size(self) -> tuple:
        """获取输出尺寸"""

        def _size_hook_fun(layer_name):
            def hook(module, input, output):
                self.layer_dict_keys_cnt += 1
                for name, m in self.model.named_modules():
                    if m is module:
                        name = f"{self.layer_dict_keys_cnt}-{name}"
                        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or
                                isinstance(m, nn.Conv3d) or isinstance(m, nn.GroupNorm)):
                            out_size = output.size()[1]
                        elif isinstance(m, nn.Linear):
                            out_size = m.out_features
                        elif isinstance(m, nn.LayerNorm):
                            out_size = m.normalized_shape[0]
                        else:
                            console.print(f"[yellow]Warning: Unknown module type: {type(m).__name__}[/yellow]")
                        self.output_sizes[name] = {"Output": [out_size], "Input": input[0].size()}
                        break

            return hook

        path = self.TOOL_LOG_FILE_PATH + "/output_sizes.pth"

        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):  # 只对 nn.Module 注册 hook
                handle = module.register_forward_hook(_size_hook_fun(name))
                self.handles.append(handle)

        if self.dataset == "ade20k":
            img = mmcv.imread("./demo/demo-ade20k.jpg")
        elif self.dataset == "cityscapes":
            img = mmcv.imread("./demo/demo.png")

        self.layer_dict_keys_cnt = 0
        inference_model(self.model, img)
        torch.save(self.output_sizes, path)
        self.layer_dict_keys_cnt = 0
        self.layer_dict_keys = list(self.output_sizes.keys())

        # 在运行结束后移除所有 hook
        for handle in self.handles:
            handle.remove()

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            assert len(input[0]) == output.shape[0], f"input has {len(input[0])}, output has {output.shape}."

            if output.size()[0] != 1:
                tmp_output = output.unsqueeze(0)
                tmp_output = tmp_output.mean((1, 2))
            else:
                # 🚀 优化：避免深拷贝，使用detach()即可
                tmp_output = output.detach()

                if len(tmp_output.size()) == 4 and not (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((2, 3))
                elif len(tmp_output.size()) == 4 and (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((1, 2))
                elif len(tmp_output.size()) == 3:  # (H, W, C)
                    tmp_output = tmp_output.mean(dim=1)
                else:
                    logger.error(f"Unexpected output shape")
                    logger.error(f"Module: {module}")
                    logger.error(f"Output size: {tmp_output.size()}")
                    logger.error(f"Dimensions: {len(tmp_output.size())}")
                    logger.error(f"Is Linear: {isinstance(module, nn.Linear)}")
                    exit()

            # check name
            for name, m in self.model.named_modules():
                if m is module:
                    name = f"{self.layer_dict_keys_cnt + 1}-{name}"
                    try:
                        if name == self.layer_dict_keys[self.layer_dict_keys_cnt]:
                            # 🚀 优化：直接保存detached tensor，避免CPU转换
                            self.layer_output_input_dict[self.layer_dict_keys[self.layer_dict_keys_cnt]] = tmp_output
                            break
                    except Exception as e:
                        logger.warning(f"{name} not in layer_dict_keys: {e}. Skipping")

            if self.output == None or not torch.equal(self.output, tmp_output):
                self.layer_dict_keys_cnt += 1

            self.output = tmp_output

        return hook

    def register_hooks(self):
        """
        注册钩子
        Register hooks
        """
        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):  # 只对 nn.Module 注册 hook
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def remove_hooks(self):
        """
        移除钩子
        Remove hooks
        """
        self.layer_output_input_dict = {}
        self.layer_dict_keys_cnt = 0
        self.save_output = None
        self.handles = []

        for handle in self.handles:
            handle.remove()

    def get_prediction(self, image):
        """
        获取模型预测结果（层输出字典，用于覆盖率计算）
        Get model prediction (layer output dict for coverage calculation)
        """
        self.layer_output_input_dict.clear()
        self.layer_dict_keys_cnt = 0
        self.output = None

        with torch.no_grad():
            result = inference_model(self.model, image)
            layer_output_dict = copy.deepcopy(self.layer_output_input_dict)
            pred_sem_seg = result.pred_sem_seg.data.cpu().numpy()

        return layer_output_dict, pred_sem_seg

    def CoverageGain(self, gain):
        if gain is not None:
            if isinstance(gain, tuple):
                return gain[0] > 0
            else:
                return gain > 0
        else:
            return False

    def _mutate(self, S: List, Ps: callable, muta_save_path: str,
                ori_save_path: str) -> List:
        """单任务并行变异"""
        I_new = []
        I_old = []
        tasks = []

        hyper_params = {
            "TRY_NUM": self.config.try_num,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
        }

        for s_i in range(len(S)):
            I = S[s_i]
            for _ in range(1, Ps(s_i) + 1):
                mutations = I.Img_Mutate(self.config.try_num, muta_save_path, ori_save_path, self.epoch)
                if mutations is None:
                    continue
                layer_output_dict, _ = self.get_prediction(mutations.img)
                _, cov_dict = self.criterion.calculate(layer_output_dict)
                gain = self.criterion.gain(cov_dict)
                if self.CoverageGain(gain) or self.epoch == 0:
                    self.criterion.update(cov_dict, gain)

                    utility.save_img(muta_save_path, self.epoch, mutations.img, mutations.mask, mutations.img_path,
                                     mutations.seg_path, mutations.image_transforms, mutations.trans_cnt)
                    utility.save_img(ori_save_path, self.epoch, mutations.ori_img, mutations.ori_mask,
                                     mutations.img_path, mutations.seg_path, mutations.image_transforms,
                                     mutations.trans_cnt, ori_pic=True)

                    I_new.append(mutations)
                    break

        return I_new

    def _initialize_logger(self):
        """初始化日志系统"""
        return get_fuzzer_logger(log_dir="./fuzzer_logs", session_name=None)

    def preprocess(self, data_list: List) -> Tuple[List, List]:
        """预处理输入数据"""
        randomize_idx = np.arange(len(data_list))
        np.random.shuffle(randomize_idx)
        images = [data_list[idx] for idx in randomize_idx]
        self.logger.log_message(f"Preprocessed {len(images)} initial images")
        return list(np.zeros(len(images))), images

    def start_new_epoch(self):
        """开始新epoch"""
        # 使用 logger 的 start_epoch，它会自动初始化统计
        self.logger.start_epoch(self.epoch)

        # 创建本地统计对象用于临时计算
        self.current_epoch_stats = EpochStats(
            epoch_num=self.epoch, start_time=time.time()
        )

    def end_current_epoch(self):
        """结束当前epoch"""
        if self.current_epoch_stats and self.current_epoch_stats.start_time:
            self.current_epoch_stats.runtime = time.time() - self.current_epoch_stats.start_time

            # 1. display_epoch_progress 会自动调用 logger 的 add_* 方法更新统计并显示进度
            display_epoch_progress(
                self.current_epoch_stats.epoch_num,
                self.current_epoch_stats.generated_count,
                self.current_epoch_stats.valid_count,
                self.current_epoch_stats.coverage_improvements,
                self.current_epoch_stats.defect_detections,
                self.current_epoch_stats.runtime,
            )

            # 2. 调用 logger.end_epoch() 保存覆盖率历史和 epoch 统计
            # skip_summary=True 避免重复输出（display_epoch_progress 已经输出了进度）
            self.logger.end_epoch(skip_summary=True)

    def update_batch_priority(self, T: Tuple, B_id):
        """更新批次优先级"""
        B_c, Bs = T
        B_c[B_id.img_idx] += 1

    def print_epoch_info(self):
        """显示当前epoch信息"""
        print("    - Current Epoch:", self.epoch)

        cov_info = self.criterion.current
        display_coverage_summary(self.cov_type,cov_info, self.old_cov_record, self.epoch)

        if self.old_cov_record is None:
            self.old_cov_record = cov_info
        else:
            self.old_cov_record = copy.deepcopy(cov_info)

        print(f"Delta time: {self.delta_time:.1f}s")

    def can_terminate(self) -> bool:
        """检查是否可以终止"""
        max_runtime_seconds = self.config.max_runtime_hours * 3600
        conditions = [
            self.epoch > self.config.max_epochs,
            self.delta_time > max_runtime_seconds,
        ]

        if any(conditions):
            if self.epoch > self.config.max_epochs:
                self.logger.log_message("Terminating: Maximum epochs reached")
            if self.delta_time > max_runtime_seconds:
                self.logger.log_message("Terminating: Maximum runtime reached")
            return True

        return False

    def compute_miou(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算mIoU
        Compute mean Intersection over Union

        Args:
            pred: 预测掩码 (H, W)
            target: 真实标签 (H, W)
            num_classes: 类别数量

        Returns:
            mIoU value
        """
        # Convert to numpy if tensor
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        ious = []
        for cls in range(self.num_classes):
            pred_mask = pred == cls
            target_mask = target == cls

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            if union == 0:
                iou = float("nan")
            else:
                iou = intersection / union
            ious.append(iou)

        ious = np.array(ious)
        miou = np.nanmean(ious)

        return miou

    def run_single_epoch(self, B: List, T: Tuple, start_time: float) -> List:
        """运行单个epoch"""
        self.start_new_epoch()
        self.print_epoch_info()

        # 执行变异
        S = B
        Ps = self.priority_scheduler.power_schedule(S, self.config.k)
        cov_save_path = f"{self.fuzz_save_path_prefix}/cov_update"
        ade_save_path = f"{self.fuzz_save_path_prefix}/ade_pic"

        ori_save_path = f"{cov_save_path}/ori"
        muta_save_path = f"{cov_save_path}/muta"

        # 执行变异
        I_new = self._mutate(S, Ps, muta_save_path, ori_save_path)

        # 检查对抗样本，以miou来计算
        for I in I_new:
            self.update_batch_priority(T, I)
            B.append(I)

            _, muta_seg = self.get_prediction(I.img)
            _, ori_seg = self.get_prediction(I.ori_img)

            muta_iou = self.compute_miou(muta_seg, I.mask)
            ori_iou = self.compute_miou(ori_seg, I.ori_mask)
            if muta_iou < ori_iou:
                utility.save_img(ade_save_path, self.epoch, I.img, I.mask, I.img_path,
                                 I.seg_path, I.image_transforms, I.trans_cnt)
                self.current_epoch_stats.defect_detections += 1

        # 更新统计信息
        self.current_epoch_stats.generated_count = 1
        self.current_epoch_stats.valid_count = len(I_new)
        self.current_epoch_stats.coverage_improvements = len(I_new)

        # 清理内存
        gc.collect()

        # 结束epoch
        self.end_current_epoch()

        # 更新全局状态
        self.epoch += 1
        self.delta_time = time.time() - start_time
        self.logger.update_runtime(self.delta_time)

        print(f"程序运行时间为： {self.delta_time:.1f}s")

        return self.priority_scheduler.select_next_batch(T, self.config.every_pic)

    def run(self, I_input: List):
        """主运行函数"""
        self.logger.log_message(f"Starting fuzzer run with {len(I_input)} input images")

        # 预处理
        T = self.preprocess(I_input)
        del I_input
        gc.collect()

        # 选择初始批次
        B = self.priority_scheduler.select_next_batch(T, self.config.every_pic)
        start_time = time.time()

        # 主循环
        while not self.can_terminate():
            B = self.run_single_epoch(B, T, start_time)

        self.logger.log_message("Fuzzer run completed")

    def exit(self):
        """退出清理"""
        self.logger.log_message("Fuzzer exit called")
        logger.info("Fuzzer exited")
        self.print_epoch_info()

        logger.info("=" * 80)
        logger.info("Fuzzer has finished running. Please check the results in the specified directory.")
        logger.info("=" * 80)

        # 完成日志记录和清理
        self.logger.finalize()
        gc.collect()
        torch.cuda.empty_cache()