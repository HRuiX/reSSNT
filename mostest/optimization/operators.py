"""
统一遗传算子 - 支持变换优化
Unified Genetic Operators - Supports transform optimization

注意：染色体的第一个基因（索引0）是图片索引，需要特殊处理：
- 交叉时：子代继承父代的图片索引
- 变异时：有小概率改变图片索引（默认5%）
"""

import numpy as np
from typing import Tuple, List, Optional

# 导入Chromosome类
try:
    from ..core.chromosome import Chromosome
except (ImportError, ValueError):
    from core.chromosome import Chromosome


class UnifiedGeneticOperators:
    """
    统一遗传算子类 - 支持"图片索引+变换"编码方案

    当前可用操作：
    1. 传统操作（与新编码方案兼容）：
       - SBX交叉 (sbx_crossover) - 跳过图片索引基因
       - 多项式变异 (polynomial_mutation) - 图片索引独立变异概率

    2. 自适应策略：
       - 自适应变异率 (adaptive_mutation_rate)

    注意：以下旧方法已被注释（与新编码方案不兼容）：
    - transform_level_crossover, transform_level_mutation
    - transform_switch_mutation, combined_mutation
    - add_transform_mutation, update_transform_mutation
    - replace_transform_mutation, remove_transform_mutation
    - incremental_mutation, _adaptive_incremental_probs

    新编码方案：[image_idx, T1_enable, T1_params, T2_enable, T2_params, ...]
    - 第一个基因（索引0）：图片索引（归一化到0-1）
    - 后续基因：变换序列（每个变换包含启用标志和参数）
    """

    def __init__(
        self,
        crossover_prob: float = 0.9,
        crossover_eta: float = 30.0,
        mutation_eta: float = 20.0,
        image_mutation_prob: float = 0.5     # 图片索引变异概率
    ):
        """
        初始化统一遗传算子

        Args:
            crossover_prob: 交叉概率
            crossover_eta: SBX分布指数
            mutation_eta: 多项式变异分布指数
            image_mutation_prob: 图片索引变异概率
        """
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.image_mutation_prob = image_mutation_prob

    # ========================================================================
    # 1. 传统交叉操作
    # ========================================================================

    def sbx_crossover(self,parent1: Chromosome,parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        模拟二进制交叉 (Simulated Binary Crossover)

        对两个父代的基因向量进行SBX交叉。

        注意：第一个基因（索引0）是图片索引，子代直接继承父代的图片索引，
        不进行交叉操作。

        Args:
            parent1, parent2: 父代染色体

        Returns:
            (child1, child2) 子代元组
        """
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        # 确保基因长度相同
        if len(genes1) != len(genes2):
            return parent1.copy(), parent2.copy()

        child1_genes = genes1.copy()
        child2_genes = genes2.copy()

        # 对变换基因进行SBX交叉（从索引1开始，跳过图片索引）
        for i in range(1, len(genes1)):
            u = np.random.rand()

            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (self.crossover_eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.crossover_eta + 1.0))

            child1_genes[i] = 0.5 * ((1.0 - beta) * genes1[i] + (1.0 + beta) * genes2[i])
            child2_genes[i] = 0.5 * ((1.0 + beta) * genes1[i] + (1.0 - beta) * genes2[i])

            # 边界约束
            child1_genes[i] = np.clip(child1_genes[i], 0.0, 1.0)
            child2_genes[i] = np.clip(child2_genes[i], 0.0, 1.0)

        # 图片索引直接继承父代（索引0）
        # child1_genes[0] 和 child2_genes[0] 已经是父代的值了

        # 创建子代
        child1 = Chromosome(
            transform_configs=parent1.transform_configs,
            genes=child1_genes,
            spatial_enabled=parent1.spatial_enabled,
            single_transform_init=False,  # 使用提供的基因，不进行单变换初始化
            num_images=parent1.num_images
        )

        child2 = Chromosome(
            transform_configs=parent2.transform_configs,
            genes=child2_genes,
            spatial_enabled=parent2.spatial_enabled,
            single_transform_init=False,  # 使用提供的基因，不进行单变换初始化
            num_images=parent2.num_images
        )

        # 强制每个子代最多保留3个启用的变换
        child1.enforce_max_transforms(max_transforms=3)
        child2.enforce_max_transforms(max_transforms=3)

        return child1, child2

    def polynomial_mutation(
        self,
        chromosome: Chromosome,
        mutation_prob: float = None
    ) -> Chromosome:
        """
        多项式变异 (Polynomial Mutation)

        对每个基因应用多项式扰动。

        注意：
        - 图片索引基因（索引0）以 image_mutation_prob 概率独立变异
        - 其他基因以 mutation_prob 概率变异

        Args:
            chromosome: 输入染色体
            mutation_prob: 变异概率（默认为1/基因数）

        Returns:
            变异后的染色体
        """
        if mutation_prob is None:
            mutation_prob = 1.0 / max(len(chromosome.genes), 1)

        mutated_genes = chromosome.genes.copy()

        if np.random.rand() < self.image_mutation_prob:
            mutated_genes[0] = np.random.uniform(0.0, 1.0)

        for i in range(1, len(mutated_genes)):
            if np.random.rand() < mutation_prob:
                u = np.random.rand()

                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (self.mutation_eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.mutation_eta + 1.0))

                mutated_genes[i] = mutated_genes[i] + delta
                mutated_genes[i] = np.clip(mutated_genes[i], 0.0, 1.0)

        mutated_chromosome = Chromosome(
            transform_configs=chromosome.transform_configs,
            genes=mutated_genes,
            spatial_enabled=chromosome.spatial_enabled,
            single_transform_init=False,  # 使用提供的基因，不进行单变换初始化
            num_images=chromosome.num_images
        )

        # 强制最多保留3个启用的变换
        mutated_chromosome.enforce_max_transforms(max_transforms=3)

        return mutated_chromosome

    def adaptive_mutation_rate(self,generation: int,max_generations: int,prob_max: float = 0.15,prob_min: float = 0.05) -> float:
        """
        自适应变异率 - 随进化进程逐渐降低

        Args:
            generation: 当前代数
            max_generations: 最大代数
            prob_max: 最大变异率
            prob_min: 最小变异率

        Returns:
            当前代的变异率
        """
        if max_generations <= 1:
            return prob_max

        progress = generation / max_generations
        mutation_rate = prob_max - (prob_max - prob_min) * progress

        return np.clip(mutation_rate, prob_min, prob_max)



def tournament_selection(population: List[Chromosome],tournament_size: int = 2) -> Chromosome:
    """
    锦标赛选择

    从种群中随机选择tournament_size个个体，返回其中最好的一个。

    比较标准（按优先级）：
    1. Rank（越小越好）- Pareto前沿
    2. Crowding distance（越大越好）- 多样性

    Args:
        population: 种群
        tournament_size: 锦标赛大小

    Returns:
        选中的染色体
    """
    indices = np.random.choice(len(population), size=tournament_size, replace=False)
    candidates = [population[i] for i in indices]

    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate.rank < best.rank:
            best = candidate
        elif candidate.rank == best.rank:
            if candidate.crowding_distance > best.crowding_distance:
                best = candidate

    return best


def binary_tournament_selection(population: List[Chromosome]) -> Chromosome:
    """
    二元锦标赛选择（tournament_size=2的特例）

    Args:
        population: 种群

    Returns:
        选中的染色体
    """
    return tournament_selection(population, tournament_size=2)
