"""
测试染色体修复
Test Chromosome Fixes

验证以下修复：
1. params 和 enabled_transforms 属性是否正确添加
2. ori_data 和 muta_data 是否可以正确赋值
3. 图片索引边界检查是否正常
4. update_params_info() 方法是否工作正常
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.chromosome import Chromosome
from core.transform_registry import TRANSFORM_CONFIGS


def test_chromosome_attributes():
    """测试染色体属性"""
    print("=" * 80)
    print("测试 1: 染色体属性")
    print("=" * 80)

    # 创建染色体
    chromosome = Chromosome(
        transform_configs=TRANSFORM_CONFIGS,
        spatial_enabled=False,
        single_transform_init=True,
        num_images=10
    )

    # 检查新属性
    assert hasattr(chromosome, 'ori_data'), "❌ 缺少 ori_data 属性"
    assert hasattr(chromosome, 'muta_data'), "❌ 缺少 muta_data 属性"
    assert hasattr(chromosome, 'params'), "❌ 缺少 params 属性"
    assert hasattr(chromosome, 'enabled_transforms'), "❌ 缺少 enabled_transforms 属性"

    print("✓ 所有新属性都存在")

    # 检查初始值
    assert chromosome.ori_data is None, "❌ ori_data 初始值应为 None"
    assert chromosome.muta_data is None, "❌ muta_data 初始值应为 None"
    assert chromosome.params is None, "❌ params 初始值应为 None"
    assert chromosome.enabled_transforms == [], "❌ enabled_transforms 初始值应为空列表"

    print("✓ 所有属性初始值正确")


def test_image_index():
    """测试图片索引"""
    print("\n" + "=" * 80)
    print("测试 2: 图片索引边界检查")
    print("=" * 80)

    num_images = 10
    chromosome = Chromosome(
        transform_configs=TRANSFORM_CONFIGS,
        spatial_enabled=False,
        single_transform_init=True,
        num_images=num_images
    )

    # 测试正常范围
    for i in range(num_images):
        chromosome.set_image_index(i)
        idx = chromosome.get_image_index()
        assert 0 <= idx < num_images, f"❌ 索引 {idx} 超出范围 [0, {num_images-1}]"

    print(f"✓ 所有正常索引 [0, {num_images-1}] 都正确")

    # 测试边界情况：基因值为 1.0
    chromosome.genes[0] = 1.0
    idx = chromosome.get_image_index()
    assert idx == num_images - 1, f"❌ 基因值=1.0时，索引应为 {num_images-1}，实际为 {idx}"

    print(f"✓ 边界情况（基因=1.0）处理正确，索引={idx}")

    # 测试边界情况：基因值为 0.0
    chromosome.genes[0] = 0.0
    idx = chromosome.get_image_index()
    assert idx == 0, f"❌ 基因值=0.0时，索引应为 0，实际为 {idx}"

    print(f"✓ 边界情况（基因=0.0）处理正确，索引={idx}")


def test_update_params_info():
    """测试 update_params_info 方法"""
    print("\n" + "=" * 80)
    print("测试 3: update_params_info() 方法")
    print("=" * 80)

    chromosome = Chromosome(
        transform_configs=TRANSFORM_CONFIGS,
        spatial_enabled=False,
        single_transform_init=True,
        num_images=10
    )

    # 调用 update_params_info
    chromosome.update_params_info()

    # 检查是否填充了信息
    assert chromosome.params is not None, "❌ params 应该被填充"
    assert isinstance(chromosome.params, dict), "❌ params 应该是字典"
    assert isinstance(chromosome.enabled_transforms, list), "❌ enabled_transforms 应该是列表"

    print(f"✓ params 字典已填充，包含 {len(chromosome.params)} 个条目")
    print(f"✓ enabled_transforms 列表包含 {len(chromosome.enabled_transforms)} 个变换")

    # 打印启用的变换
    if chromosome.enabled_transforms:
        print(f"  启用的变换: {', '.join(chromosome.enabled_transforms)}")


def test_data_assignment():
    """测试 ori_data 和 muta_data 赋值"""
    print("\n" + "=" * 80)
    print("测试 4: ori_data 和 muta_data 赋值")
    print("=" * 80)

    chromosome = Chromosome(
        transform_configs=TRANSFORM_CONFIGS,
        spatial_enabled=False,
        single_transform_init=True,
        num_images=10
    )

    # 赋值测试数据
    test_ori_data = {
        "seed_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "seed_mask": np.random.randint(0, 19, (224, 224), dtype=np.uint8),
        "file_name": "test_image.jpg"
    }

    test_muta_data = {
        "muta_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "muta_mask": np.random.randint(0, 19, (224, 224), dtype=np.uint8),
        "file_name": "test_image_mutated.jpg"
    }

    chromosome.ori_data = test_ori_data
    chromosome.muta_data = test_muta_data

    # 检查赋值是否成功
    assert chromosome.ori_data is not None, "❌ ori_data 赋值失败"
    assert chromosome.muta_data is not None, "❌ muta_data 赋值失败"
    assert chromosome.ori_data["file_name"] == "test_image.jpg", "❌ ori_data 内容错误"
    assert chromosome.muta_data["file_name"] == "test_image_mutated.jpg", "❌ muta_data 内容错误"

    print("✓ ori_data 赋值成功")
    print("✓ muta_data 赋值成功")


def test_copy():
    """测试 copy 方法"""
    print("\n" + "=" * 80)
    print("测试 5: copy() 方法")
    print("=" * 80)

    # 创建染色体并设置数据
    chromosome = Chromosome(
        transform_configs=TRANSFORM_CONFIGS,
        spatial_enabled=False,
        single_transform_init=True,
        num_images=10
    )

    chromosome.ori_data = {"file_name": "test.jpg"}
    chromosome.muta_data = {"file_name": "test_mutated.jpg"}
    chromosome.update_params_info()

    # 复制
    copied = chromosome.copy()

    # 检查所有属性是否正确复制
    assert copied.ori_data == chromosome.ori_data, "❌ ori_data 复制失败"
    assert copied.muta_data == chromosome.muta_data, "❌ muta_data 复制失败"
    assert copied.params == chromosome.params, "❌ params 复制失败"
    assert copied.enabled_transforms == chromosome.enabled_transforms, "❌ enabled_transforms 复制失败"

    print("✓ 所有新属性都正确复制")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始测试染色体修复")
    print("=" * 80 + "\n")

    try:
        test_chromosome_attributes()
        test_image_index()
        test_update_params_info()
        test_data_assignment()
        test_copy()

        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
