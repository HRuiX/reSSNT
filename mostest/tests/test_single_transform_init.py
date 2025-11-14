"""
测试单变换初始化功能
Test Single Transform Initialization Feature
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mostest.core.chromosome import Chromosome
from mostest.core.transform_configs import get_balanced_config, get_lightweight_config
from mostest.optimization.operators import GeneticOperators


def test_single_transform_initialization():
    """测试单变换初始化"""
    print("=" * 80)
    print("测试1: 单变换初始化")
    print("=" * 80)

    configs = get_balanced_config(fixed_mode=True)

    # 创建多个染色体,验证每个只启用一个变换
    num_samples = 10
    chromosomes = []

    print(f"\n创建 {num_samples} 个染色体,每个应该只启用1个变换:")
    print("-" * 60)

    for i in range(num_samples):
        chrom = Chromosome(configs, spatial_enabled=False, single_transform_init=True)
        chromosomes.append(chrom)

        # 统计启用的变换数量
        enabled_transforms = []
        for j in range(chrom.num_transforms):
            start_idx, _ = chrom.get_gene_index(j)
            if chrom.genes[start_idx] >= 0.5:
                enabled_transforms.append(j)

        enabled_count = len(enabled_transforms)

        # 获取启用的变换名称
        summary = chrom.get_transform_summary()
        enabled_names = [s['name'] for s in summary]

        status = "✓" if enabled_count == 1 else "✗"
        print(f"  染色体{i+1}: 启用 {enabled_count} 个变换 {status}")
        if enabled_names:
            print(f"           -> {', '.join(enabled_names)}")

    # 验证所有染色体都只启用了一个变换
    all_single = all(len(chrom.get_transform_summary()) == 1 for chrom in chromosomes)

    if all_single:
        print("\n✓ 所有染色体都只启用了一个变换!")
    else:
        print("\n✗ 部分染色体启用了多个变换!")

    return all_single


def test_transform_distribution():
    """测试变换分布的均匀性"""
    print("\n" + "=" * 80)
    print("测试2: 变换分布均匀性")
    print("=" * 80)

    configs = get_balanced_config(fixed_mode=True)
    num_samples = 100

    # 统计每个变换被启用的次数
    transform_counts = {i: 0 for i in range(len(configs))}

    for _ in range(num_samples):
        chrom = Chromosome(configs, spatial_enabled=False, single_transform_init=True)

        for i in range(chrom.num_transforms):
            start_idx, _ = chrom.get_gene_index(i)
            if chrom.genes[start_idx] >= 0.5:
                transform_counts[i] += 1

    print(f"\n在 {num_samples} 个染色体中,每个变换被启用的次数:")
    print("-" * 60)

    for i, config in enumerate(configs):
        count = transform_counts[i]
        percentage = (count / num_samples) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {config.name:25s}: {count:3d} 次 ({percentage:5.1f}%) {bar}")

    # 计算分布的标准差(理想情况下应该接近均匀分布)
    expected = num_samples / len(configs)
    counts = list(transform_counts.values())
    std_dev = np.std(counts)

    print(f"\n统计信息:")
    print(f"  期望值: {expected:.1f}")
    print(f"  标准差: {std_dev:.2f}")
    print(f"  变异系数: {(std_dev/expected)*100:.1f}%")

    # 变异系数小于50%认为分布较均匀
    is_uniform = (std_dev/expected) < 0.5

    if is_uniform:
        print("\n✓ 变换分布较为均匀!")
    else:
        print("\n⚠ 变换分布可能不够均匀")

    return is_uniform


def test_genetic_operations_enable_multiple():
    """测试遗传操作后可以启用多个变换"""
    print("\n" + "=" * 80)
    print("测试3: 遗传操作后启用多个变换")
    print("=" * 80)

    configs = get_lightweight_config(fixed_mode=True)
    genetic_ops = GeneticOperators()

    # 创建两个初始染色体(各启用一个变换)
    parent1 = Chromosome(configs, spatial_enabled=False, single_transform_init=True)
    parent2 = Chromosome(configs, spatial_enabled=False, single_transform_init=True)

    print("\n父代:")
    summary1 = parent1.get_transform_summary()
    summary2 = parent2.get_transform_summary()
    print(f"  父代1: 启用 {len(summary1)} 个变换 - {[s['name'] for s in summary1]}")
    print(f"  父代2: 启用 {len(summary2)} 个变换 - {[s['name'] for s in summary2]}")

    # 交叉
    child1, child2 = genetic_ops.sbx_crossover(parent1, parent2)

    summary_c1 = child1.get_transform_summary()
    summary_c2 = child2.get_transform_summary()

    print(f"\n交叉后:")
    print(f"  子代1: 启用 {len(summary_c1)} 个变换 - {[s['name'] for s in summary_c1]}")
    print(f"  子代2: 启用 {len(summary_c2)} 个变换 - {[s['name'] for s in summary_c2]}")

    # 变异
    mutated = genetic_ops.polynomial_mutation(child1, mutation_prob=0.2)
    summary_m = mutated.get_transform_summary()

    print(f"\n变异后:")
    print(f"  变异个体: 启用 {len(summary_m)} 个变换 - {[s['name'] for s in summary_m]}")

    # 多次交叉和变异,统计启用变换数量的分布
    print("\n" + "-" * 60)
    print("模拟多代进化,统计启用的变换数量:")
    print("-" * 60)

    enabled_count_distribution = {0: 0, 1: 0, 2: 0, 3: 0}

    # 创建初始种群
    population = [Chromosome(configs, spatial_enabled=False, single_transform_init=True)
                  for _ in range(20)]

    # 进行5代进化
    for generation in range(5):
        new_population = []
        for _ in range(len(population)):
            # 选择父代
            p1 = population[np.random.randint(len(population))]
            p2 = population[np.random.randint(len(population))]

            # 交叉
            c1, c2 = genetic_ops.sbx_crossover(p1, p2)

            # 变异(使用较高的变异率)
            m1 = genetic_ops.polynomial_mutation(c1, mutation_prob=0.3)

            new_population.append(m1)

        population = new_population

    # 统计最后一代的启用变换数量
    for ind in population:
        count = len(ind.get_transform_summary())
        if count <= 3:
            enabled_count_distribution[count] += 1
        else:
            if count not in enabled_count_distribution:
                enabled_count_distribution[count] = 0
            enabled_count_distribution[count] += 1

    for count, freq in sorted(enabled_count_distribution.items()):
        bar = "█" * (freq // 2)
        print(f"  启用 {count} 个变换: {freq:3d} 次 {bar}")

    # 检查是否有多变换启用的情况
    has_multiple = sum(v for k, v in enabled_count_distribution.items() if k > 1) > 0

    if has_multiple:
        print("\n✓ 遗传操作后可以启用多个变换!")
    else:
        print("\n⚠ 遗传操作后仍然只有单个变换启用")

    return has_multiple


def test_disable_single_transform_init():
    """测试禁用单变换初始化"""
    print("\n" + "=" * 80)
    print("测试4: 禁用单变换初始化(single_transform_init=False)")
    print("=" * 80)

    configs = get_balanced_config(fixed_mode=True)

    # 创建多个染色体,不限制为单变换
    num_samples = 10

    print(f"\n创建 {num_samples} 个染色体(允许多变换启用):")
    print("-" * 60)

    enabled_counts = []

    for i in range(num_samples):
        chrom = Chromosome(configs, spatial_enabled=False, single_transform_init=False)

        summary = chrom.get_transform_summary()
        enabled_count = len(summary)
        enabled_counts.append(enabled_count)

        enabled_names = [s['name'] for s in summary]
        print(f"  染色体{i+1}: 启用 {enabled_count} 个变换")
        if enabled_names:
            print(f"           -> {', '.join(enabled_names)}")

    avg_enabled = np.mean(enabled_counts)

    print(f"\n统计:")
    print(f"  平均启用变换数: {avg_enabled:.2f}")
    print(f"  范围: [{min(enabled_counts)}, {max(enabled_counts)}]")

    # 应该有一些染色体启用了多个变换
    has_multiple = any(count > 1 for count in enabled_counts)

    if has_multiple:
        print("\n✓ 禁用单变换初始化后,可以启用多个变换!")
    else:
        print("\n⚠ 即使禁用单变换初始化,也没有启用多个变换")

    return has_multiple


def test_population_creation():
    """测试种群创建"""
    print("\n" + "=" * 80)
    print("测试5: 种群创建")
    print("=" * 80)

    configs = get_balanced_config(fixed_mode=True)
    population_size = 50

    # 使用单变换初始化
    population = Chromosome.create_random_population(
        population_size=population_size,
        transform_configs=configs,
        spatial_enabled=False,
        single_transform_init=True
    )

    print(f"\n创建种群 (size={population_size}, single_transform_init=True):")
    print("-" * 60)

    # 统计启用变换数量
    enabled_counts = []
    for ind in population:
        count = len(ind.get_transform_summary())
        enabled_counts.append(count)

    all_single = all(count == 1 for count in enabled_counts)

    print(f"  所有个体都只启用1个变换: {'是 ✓' if all_single else '否 ✗'}")
    print(f"  启用变换数分布: {dict(zip(*np.unique(enabled_counts, return_counts=True)))}")

    return all_single


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("单变换初始化功能 - 完整测试")
    print("=" * 80)

    results = []

    try:
        results.append(("单变换初始化", test_single_transform_initialization()))
        results.append(("变换分布均匀性", test_transform_distribution()))
        results.append(("遗传操作多变换", test_genetic_operations_enable_multiple()))
        results.append(("禁用单变换初始化", test_disable_single_transform_init()))
        results.append(("种群创建", test_population_creation()))

        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)

        for name, result in results:
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {name:20s}: {status}")

        all_passed = all(result for _, result in results)

        if all_passed:
            print("\n" + "=" * 80)
            print("✓✓✓ 所有测试通过! ✓✓✓")
            print("=" * 80)
            print("\n功能说明:")
            print("  ✓ 初始化时只启用一个随机变换")
            print("  ✓ 变换选择分布均匀")
            print("  ✓ 遗传操作后可以启用多个变换")
            print("  ✓ 可以选择禁用单变换初始化")
            print("  ✓ 种群创建正常工作")
        else:
            print("\n" + "=" * 80)
            print("✗✗✗ 部分测试失败 ✗✗✗")
            print("=" * 80)

        return all_passed

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗✗✗ 测试出错: {e} ✗✗✗")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
