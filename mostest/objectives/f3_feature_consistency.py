"""
F3: 特征-性能一致性度量
"""

import torch
import numpy as np
from typing import Dict
from rich.console import Console
console = Console()

class F3FeatureConsistency:
    """
    F3 Objective: Consistency between internal feature changes and output performance changes
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1e-8  # 防止除零错误 Prevent division by zero

    def compute_feature_change(
        self,
        layer_output_orig: Dict[str, torch.Tensor],
        layer_output_mut: Dict[str, torch.Tensor]
    ) -> float:
        feature_changes = []

        for layer_name in layer_output_orig.keys():
            if layer_name not in layer_output_mut:
                continue

            feat_orig = layer_output_orig[layer_name]
            feat_mut = layer_output_mut[layer_name]

            if not isinstance(feat_orig, torch.Tensor) or not isinstance(feat_mut, torch.Tensor):
                continue

            diff_norm = torch.norm(feat_orig - feat_mut, p=2)
            orig_norm = torch.norm(feat_orig, p=2)

            normalized_change = (diff_norm / (orig_norm + self.epsilon)).item()
            feature_changes.append(normalized_change)

        if not feature_changes:
            return 0.0

        return float(np.mean(feature_changes))

    def compute_iou(self, pred: np.ndarray, label: np.ndarray) -> float:
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(label):
            label = label.cpu().numpy()

        valid_mask = (label != 255)
        correct_mask = (pred == label) & valid_mask

        intersection = np.sum(correct_mask)
        union = np.sum(valid_mask)

        if union == 0:
            return 1.0

        return float(intersection / union)

    def compute_performance_change(
        self,
        pred_orig: np.ndarray,
        label_orig: np.ndarray,
        pred_mut: np.ndarray,
        label_mut: np.ndarray
    ) -> float:
        iou_orig = self.compute_iou(pred_orig, label_orig)
        iou_mut = self.compute_iou(pred_mut, label_mut)

        if iou_orig < 1e-8:
            return 0.0

        normalized_change = abs(iou_orig - iou_mut) / iou_orig

        return float(normalized_change)

    def compute(
        self,
        ori_pred_sem_seg: np.ndarray,
        seed_mask: np.ndarray,
        muta_pred_sem_seg: np.ndarray,
        muta_mask: np.ndarray,
        ori_layer_output_dict: Dict = None,
        muta_layer_output_dict: Dict = None,
        **kwargs
    ) -> float:
        delta_feature = self.compute_feature_change(ori_layer_output_dict, muta_layer_output_dict)

        delta_performance = self.compute_performance_change(
            ori_pred_sem_seg, seed_mask, muta_pred_sem_seg, muta_mask
        )

        f3 = abs(delta_feature - delta_performance)

        return float(f3)