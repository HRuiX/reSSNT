"""
快速测试脚本 - 验证MOSTest框架的各个组件
Quick Test Script - Verify MOSTest framework components
"""

import torch
import numpy as np
import sys
import os

# 添加父目录到路径以支持导入 Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_transform_registry():
    """测试变换注册表 Test transform registry"""
    print("\n" + "="*60)
    print("测试1: 变换注册表 Test 1: Transform Registry")
    print("="*60)

    from core.transform_registry import TransformRegistry, TRANSFORM_CONFIGS

    # 列出所有变换 List all transforms
    transforms = TransformRegistry.list_transforms()
    print(f"注册的变换数量 Number of registered transforms: {len(transforms)}")

    for tid, name in transforms.items():
        config = TransformRegistry.get_config(tid)
        print(f"  [{tid}] {name:25s} - 参数数量: {config.num_params}, 语义保持性: {config.semantic_preservation}")

    # 测试解码 Test decoding
    config = TransformRegistry.get_config(1)  # ShiftScaleRotate
    params = np.random.rand(config.num_params)
    decoded = config.decode_params(params)
    print(f"\n示例解码 Example decode (ShiftScaleRotate):")
    print(f"  归一化参数 Normalized params: {params}")
    print(f"  解码结果 Decoded params: {decoded}")

    print("✓ 变换注册表测试通过 Transform registry test passed")


def test_chromosome():
    """测试染色体编码 Test chromosome encoding"""
    print("\n" + "="*60)
    print("测试2: 染色体编码 Test 2: Chromosome Encoding")
    print("="*60)

    from core.chromosome import Chromosome
    from core.transform_registry import TransformRegistry

    # 创建染色体 Create chromosome
    transform_seq = TransformRegistry.get_default_sequence()
    chromosome = Chromosome(transform_seq, num_images=10)  # 假设有10张图片

    print(f"变换序列 Transform sequence: {transform_seq}")
    print(f"染色体长度 Chromosome length: {len(chromosome.genes)}")
    print(f"图片索引 Image index: {chromosome.get_image_index()}")
    print(f"基因范围 Gene range: [{chromosome.genes.min():.3f}, {chromosome.genes.max():.3f}]")

    # 测试解码 Test decoding
    transform = chromosome.decode()
    print(f"解码的变换数量 Number of decoded transforms: {len(transform.transforms)}")

    # 测试应用 Test application
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 10, (256, 256), dtype=np.uint8)

    mutated_image, mutated_mask = chromosome.apply_transform(test_image, test_mask)

    print(f"原始图像形状 Original image shape: {test_image.shape}")
    print(f"变异图像形状 Mutated image shape: {mutated_image.shape}")
    print(f"原始掩码形状 Original mask shape: {test_mask.shape}")
    print(f"变异掩码形状 Mutated mask shape: {mutated_mask.shape}")

    print("✓ 染色体编码测试通过 Chromosome encoding test passed")


def test_genetic_operators():
    """测试遗传算子 Test genetic operators"""
    print("\n" + "="*60)
    print("测试3: 遗传算子 Test 3: Genetic Operators")
    print("="*60)

    from core.chromosome import Chromosome
    from core.transform_registry import TransformRegistry
    from optimization.genetic_operators import GeneticOperators

    # 创建遗传算子 Create genetic operators
    gen_ops = GeneticOperators()

    # 创建父代 Create parents
    transform_seq = TransformRegistry.get_default_sequence()
    parent1 = Chromosome(transform_seq, num_images=10)
    parent2 = Chromosome(transform_seq, num_images=10)

    # 测试交叉 Test crossover
    child1, child2 = gen_ops.sbx_crossover(parent1, parent2)
    print(f"交叉前父代1基因和 Parent1 gene sum before crossover: {parent1.genes.sum():.2f}")
    print(f"交叉前父代2基因和 Parent2 gene sum before crossover: {parent2.genes.sum():.2f}")
    print(f"交叉后子代1基因和 Child1 gene sum after crossover: {child1.genes.sum():.2f}")
    print(f"交叉后子代2基因和 Child2 gene sum after crossover: {child2.genes.sum():.2f}")

    # 测试变异 Test mutation
    mutated = gen_ops.polynomial_mutation(child1)
    print(f"变异前基因和 Gene sum before mutation: {child1.genes.sum():.2f}")
    print(f"变异后基因和 Gene sum after mutation: {mutated.genes.sum():.2f}")

    # 测试自适应变异率 Test adaptive mutation rate
    rates = [gen_ops.adaptive_mutation_rate(g, 100) for g in range(0, 101, 20)]
    print(f"自适应变异率 Adaptive mutation rates: {rates}")

    print("✓ 遗传算子测试通过 Genetic operators test passed")


def test_reference_points():
    """测试参考点生成 Test reference points generation"""
    print("\n" + "="*60)
    print("测试4: 参考点生成 Test 4: Reference Points Generation")
    print("="*60)

    from optimization.reference_points import generate_reference_points, associate_to_reference_point

    # 生成参考点 Generate reference points
    ref_points = generate_reference_points(num_objectives=3, divisions=4)

    print(f"参考点数量 Number of reference points: {len(ref_points)}")
    print(f"参考点形状 Reference points shape: {ref_points.shape}")
    print(f"前5个参考点 First 5 reference points:")
    for i in range(min(5, len(ref_points))):
        print(f"  {i}: {ref_points[i]}")

    # 测试关联 Test association
    test_objectives = np.random.rand(10, 3)
    associations = associate_to_reference_point(test_objectives, ref_points)

    print(f"\n关联结果 Association results:")
    print(f"  关联数组形状 Association array shape: {associations.shape}")
    print(f"  关联到的参考点索引 Associated reference point indices: {associations}")

    print("✓ 参考点生成测试通过 Reference points generation test passed")


def test_nsga3():
    """测试NSGA-III算法 Test NSGA-III algorithm"""
    print("\n" + "="*60)
    print("测试5: NSGA-III算法 Test 5: NSGA-III Algorithm")
    print("="*60)

    from core.chromosome import Chromosome
    from core.transform_registry import TransformRegistry
    from optimization.nsga3 import NSGA3

    # 创建优化器 Create optimizer
    optimizer = NSGA3(population_size=20, max_generations=10)

    # 创建测试种群 Create test population
    transform_seq = TransformRegistry.get_default_sequence()
    population = Chromosome.create_random_population(20, transform_seq, num_images=10)

    # 随机分配目标值 Randomly assign objective values
    for ind in population:
        ind.objectives = np.random.rand(3).tolist()

    print(f"初始种群大小 Initial population size: {len(population)}")

    # 测试非支配排序 Test non-dominated sorting
    fronts = optimizer.fast_non_dominated_sort(population)
    print(f"Pareto前沿数量 Number of Pareto fronts: {len(fronts)}")
    print(f"第一前沿大小 First front size: {len(fronts[0])}")

    # 测试子代生成 Test offspring generation
    offspring = optimizer.create_offspring(population, generation=0)
    print(f"子代种群大小 Offspring population size: {len(offspring)}")

    # 测试环境选择 Test environmental selection
    combined = population + offspring
    selected = optimizer.environmental_selection(combined, 20)
    print(f"选择后种群大小 Selected population size: {len(selected)}")

    print("✓ NSGA-III算法测试通过 NSGA-III algorithm test passed")


def test_coverage_metrics():
    """测试覆盖率指标 Test coverage metrics"""
    print("\n" + "="*60)
    print("测试6: 覆盖率指标 Test 6: Coverage Metrics")
    print("="*60)

    from metrics.coverage.boundary import BoundaryCoverage

    # 测试边界覆盖率 Test boundary coverage
    bc = BoundaryCoverage(num_classes=10)

    # 创建测试掩码 Create test mask
    test_mask = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
    bc.update(test_mask)

    coverage = bc.get_coverage()
    print(f"边界覆盖率 Boundary coverage: {coverage:.3f}")
    print(f"覆盖的边界对数量 Number of covered boundary pairs: {len(bc.covered_boundaries)}")
    print(f"总可能边界对数量 Total possible boundary pairs: {bc.total_possible}")

    # 再更新几次 Update a few more times
    for _ in range(5):
        test_mask = np.random.randint(0, 10, (100, 100), dtype=np.uint8)
        bc.update(test_mask)

    final_coverage = bc.get_coverage()
    print(f"更新后的覆盖率 Coverage after updates: {final_coverage:.3f}")

    print("✓ 覆盖率指标测试通过 Coverage metrics test passed")


def test_quality_metrics():
    """测试质量度量 Test quality metrics"""
    print("\n" + "="*60)
    print("测试7: 质量度量 Test 7: Quality Metrics")
    print("="*60)

    from metrics.quality import QualityMetrics

    # 创建质量度量对象 Create quality metrics object
    qm = QualityMetrics(device='cpu')

    # 创建测试图像 Create test leftImg8bit
    image1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # 稍微修改创建image2 Slightly modify to create image2
    image2 = np.clip(image1.astype(np.int32) + np.random.randint(-20, 20, image1.shape), 0, 255).astype(np.uint8)

    # 计算SSIM Compute SSIM
    ssim_val = qm.compute_ssim(image1, image2)
    print(f"SSIM: {ssim_val:.4f}")

    # 计算PSNR Compute PSNR
    psnr_val = qm.compute_psnr(image1, image2)
    print(f"PSNR: {psnr_val:.2f} dB")

    # 计算所有指标 Compute all metrics
    all_metrics = qm.compute_all(image1, image2)
    print(f"所有指标 All metrics: {all_metrics}")

    # 测试质量约束 Test quality constraints
    passed, metrics = qm.check_quality_constraints(image1, image2)
    print(f"质量约束检查 Quality constraints check: {'通过 Passed' if passed else '未通过 Failed'}")

    print("✓ 质量度量测试通过 Quality metrics test passed")


def main():
    """主测试函数 Main test function"""
    print("\n" + "="*80)
    print("MOSTest 组件测试 MOSTest Component Testing")
    print("="*80)

    try:
        test_transform_registry()
        test_chromosome()
        test_genetic_operators()
        test_reference_points()
        test_nsga3()
        test_coverage_metrics()
        test_quality_metrics()

        print("\n" + "="*80)
        print("✓ 所有测试通过！All tests passed!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ 测试失败 Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
