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

    def __init__(
        self,
        max_transforms: int,
        crossover_prob: float = 0.9,
        crossover_eta: float = 30.0,
        mutation_eta: float = 20.0,
        image_mutation_prob: float = 0.5
    ):

        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.image_mutation_prob = image_mutation_prob
        self.max_transforms = max_transforms

    def sbx_crossover(self,parent1: Chromosome,parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Simulated Binary Crossover
        """
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        if len(genes1) != len(genes2):
            return parent1.copy(), parent2.copy()

        child1_genes = genes1.copy()
        child2_genes = genes2.copy()

        for i in range(1, len(genes1)):
            u = np.random.rand()

            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (self.crossover_eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.crossover_eta + 1.0))

            child1_genes[i] = 0.5 * ((1.0 - beta) * genes1[i] + (1.0 + beta) * genes2[i])
            child2_genes[i] = 0.5 * ((1.0 + beta) * genes1[i] + (1.0 - beta) * genes2[i])

            child1_genes[i] = np.clip(child1_genes[i], 0.0, 1.0)
            child2_genes[i] = np.clip(child2_genes[i], 0.0, 1.0)

        child1 = Chromosome(
            transform_configs=parent1.transform_configs,
            genes=child1_genes,
            spatial_enabled=parent1.spatial_enabled,
            single_transform_init=False,
            num_images=parent1.num_images
        )

        child2 = Chromosome(
            transform_configs=parent2.transform_configs,
            genes=child2_genes,
            spatial_enabled=parent2.spatial_enabled,
            single_transform_init=False,
            num_images=parent2.num_images
        )

        child1.enforce_max_transforms(max_transforms=self.max_transforms)
        child2.enforce_max_transforms(max_transforms=self.max_transforms)

        return child1, child2

    def polynomial_mutation(
        self,
        chromosome: Chromosome,
        mutation_prob: float = None
    ) -> Chromosome:
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
            single_transform_init=False,
            num_images=chromosome.num_images
        )

        mutated_chromosome.enforce_max_transforms(max_transforms=self.max_transforms)

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
