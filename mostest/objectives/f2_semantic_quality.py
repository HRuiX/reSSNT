"""
F2: Prediction Quality Degradation Metric
"""

import torch
import numpy as np
from typing import Tuple, Dict
import cv2
from rich.console import Console
from functools import lru_cache

console = Console()


class F2SemanticQuality:
    """
    F2 Objective: Pixel-level, Layout, Boundary Prediction Error Changes
    """

    def __init__(self):
        self._boundary_kernel = np.ones((3, 3), np.uint8)

    def compute_pixel_error(self, pred: np.ndarray, label: np.ndarray) -> float:
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(label):
            label = label.cpu().numpy()
        return float(np.mean(pred != label))

    def compute_miou(self, pred: np.ndarray, target: np.ndarray, num_classes: int = None) -> float:
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        ious = []
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)

            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)

            if union == 0:
                iou = float('nan')
            else:
                iou = intersection / union
            ious.append(iou)

        ious = np.array(ious)
        miou = float(np.nanmean(ious))

        return miou

    def extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        mask_uint8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
        dilated = cv2.dilate(mask_uint8, self._boundary_kernel, iterations=1)
        eroded = cv2.erode(mask_uint8, self._boundary_kernel, iterations=1)
        boundary = (dilated != eroded)

        return boundary

    def compute_boundary_error(self, pred: np.ndarray, label: np.ndarray) -> float:
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(label):
            label = label.cpu().numpy()

        boundary_mask = self.extract_boundary(label)
        boundary_sum = boundary_mask.sum()

        if boundary_sum == 0:
            return 0.0

        errors = np.sum((pred != label) & boundary_mask)
        boundary_error = float(errors / boundary_sum)

        return boundary_error

    def compute_delta_pixel(
            self,
            pred_orig: np.ndarray,
            label_orig: np.ndarray,
            pred_mut: np.ndarray,
            label_mut: np.ndarray
    ) -> float:
        """Compute pixel-level error change"""
        error_orig = self.compute_pixel_error(pred_orig, label_orig)
        error_mut = self.compute_pixel_error(pred_mut, label_mut)
        return abs(error_mut - error_orig)

    def compute_delta_layout(
            self,
            pred_orig: np.ndarray,
            label_orig: np.ndarray,
            pred_mut: np.ndarray,
            label_mut: np.ndarray,
            num_classes: int = None
    ) -> float:
        """Compute layout error change (based on mIoU)"""
        miou_orig = self.compute_miou(pred_orig, label_orig, num_classes)
        miou_mut = self.compute_miou(pred_mut, label_mut, num_classes)

        error_orig = 1 - miou_orig
        error_mut = 1 - miou_mut

        return abs(error_mut - error_orig)

    def compute_delta_boundary(
            self,
            pred_orig: np.ndarray,
            label_orig: np.ndarray,
            pred_mut: np.ndarray,
            label_mut: np.ndarray
    ) -> float:
        """Compute boundary error change"""
        error_orig = self.compute_boundary_error(pred_orig, label_orig)
        error_mut = self.compute_boundary_error(pred_mut, label_mut)
        return abs(error_mut - error_orig)

    def compute(
            self,
            seed_mask: np.ndarray,
            ori_pred_sem_seg: np.ndarray,
            muta_mask: np.ndarray,
            muta_pred_sem_seg: np.ndarray,
            num_classes: int = None,
            **kwargs
    ) -> float:
        """
        Compute complete F2 objective function value
        """
        delta_pixel = self.compute_delta_pixel(ori_pred_sem_seg, seed_mask, muta_pred_sem_seg, muta_mask)
        delta_layout = self.compute_delta_layout(ori_pred_sem_seg, seed_mask, muta_pred_sem_seg, muta_mask, num_classes)
        delta_boundary = self.compute_delta_boundary(ori_pred_sem_seg, seed_mask, muta_pred_sem_seg, muta_mask)

        f2 = (delta_pixel * delta_layout * delta_boundary) ** (1 / 3)

        return float(f2)