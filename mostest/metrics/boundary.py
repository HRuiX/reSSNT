"""
Semantic Boundary Coverage (SBC)
"""

import numpy as np
from typing import Set, Tuple
from collections import defaultdict


class BoundaryCoverage:
    """
    语义边界覆盖率: 测试集覆盖了多少类别边界组合
    Semantic Boundary Coverage: Proportion of class boundary combinations covered

    公式: SBC = |{(ci, cj) : ∃x∈T, (ci,cj)∈B(x)}| / C(C, 2)
    """

    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: 类别数量
        """
        self.num_classes = num_classes

        # 存储已覆盖的边界对
        self.covered_boundaries: Set[Tuple[int, int]] = set()

        # 总可能的边界对数 Total possible boundary pairs
        self.total_possible = num_classes * (num_classes - 1) // 2

    def extract_boundary_pairs(self, mask: np.ndarray) -> Set[Tuple[int, int]]:
        """
        从分割掩码中提取边界类别对
        Extract boundary class pairs from segmentation mask

        优化：使用numpy向量化操作代替循环

        Args:
            mask: 分割掩码 (H, W)

        Returns:
            Set of (class_i, class_j) tuples where i < j
        """
        boundary_pairs = set()

        H, W = mask.shape

        # 向量化检查水平相邻像素 Vectorized horizontal neighbors
        horizontal_c1 = mask[:, :-1]
        horizontal_c2 = mask[:, 1:]
        h_diff_mask = (horizontal_c1 != horizontal_c2) & \
                      (horizontal_c1 < self.num_classes) & \
                      (horizontal_c2 < self.num_classes)

        h_indices = np.where(h_diff_mask)
        for idx in zip(h_indices[0], h_indices[1]):
            c1, c2 = horizontal_c1[idx], horizontal_c2[idx]
            pair = tuple(sorted([int(c1), int(c2)]))
            boundary_pairs.add(pair)

        # 向量化检查垂直相邻像素 Vectorized vertical neighbors
        vertical_c1 = mask[:-1, :]
        vertical_c2 = mask[1:, :]
        v_diff_mask = (vertical_c1 != vertical_c2) & \
                      (vertical_c1 < self.num_classes) & \
                      (vertical_c2 < self.num_classes)

        v_indices = np.where(v_diff_mask)
        for idx in zip(v_indices[0], v_indices[1]):
            c1, c2 = vertical_c1[idx], vertical_c2[idx]
            pair = tuple(sorted([int(c1), int(c2)]))
            boundary_pairs.add(pair)

        return boundary_pairs

    def update(self, mask: np.ndarray):
        """
        更新边界覆盖率
        Update boundary coverage

        Args:
            mask: 分割掩码 (H, W)
        """
        boundary_pairs = self.extract_boundary_pairs(mask)
        self.covered_boundaries.update(boundary_pairs)

    def get_coverage(self) -> float:
        """
        计算边界覆盖率
        Compute boundary coverage

        Returns:
            覆盖率 [0, 1]
        """
        if self.total_possible == 0:
            return 1.0

        coverage = len(self.covered_boundaries) / self.total_possible

        return coverage

    def reset(self):
        """重置覆盖率统计 Reset coverage statistics"""
        self.covered_boundaries = set()

    def get_uncovered_boundaries(self) -> Set[Tuple[int, int]]:
        """
        获取未覆盖的边界对
        Get uncovered boundary pairs

        Returns:
            Set of uncovered (class_i, class_j) pairs
        """
        all_possible = set()
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                all_possible.add((i, j))

        uncovered = all_possible - self.covered_boundaries

        return uncovered
