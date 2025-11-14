"""
使用示例：如何使用预设配置创建染色体
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from mostest.encoding.enhanced_transform_registry import TransformConfig, ParamConfig
from mostest.encoding.chromosome_enhanced import EnhancedChromosome
from mostest.encoding.default_configs import (
    get_balanced_config,
    get_lightweight_config,
    get_comprehensive_config,
    get_semantic_preserving_config,
    get_adversarial_config,
    get_configs_from_sequence
)


print("=" * 80)
print("染色体使用示例")
print("=" * 80)


# ==================== 示例1: 使用平衡型配置（推荐） ====================

print("\n【示例1】使用平衡型配置:")
print("-" * 80)

# 创建配置（启用固定参数模式）
configs = get_balanced_config(fixed_mode=True)

# 创建染色体（不启用空间定位）
chromosome_basic = EnhancedChromosome(configs, spatial_enabled=False)

print(f"变换数量: {len(configs)}")
print(f"总基因数: {chromosome_basic.total_genes}")
print(f"基因数组: {chromosome_basic.genes.shape}")

# 查看每个变换的基因数
print("\n变换详情:")
for i, (config, gene_count) in enumerate(zip(configs, chromosome_basic.genes_per_transform)):
    print(f"  {i+1}. {config.name:25s} 需要 {gene_count} 个基因")

# 获取变换摘要
print("\n解码摘要:")
summary = chromosome_basic.get_transform_summary()
for i, info in enumerate(summary, 1):
    print(f"\n  变换 {i}: {info['name']}")
    print(f"    启用: {info['enabled']}")
    print(f"    参数: {info['params']}")


# ==================== 示例2: 使用空间定位功能 ====================

print("\n\n【示例2】启用空间定位:")
print("-" * 80)

# 创建配置
configs = get_lightweight_config(fixed_mode=True)

# 创建染色体（启用空间定位）
chromosome_spatial = EnhancedChromosome(configs, spatial_enabled=True)

print(f"变换数量: {len(configs)}")
print(f"总基因数: {chromosome_spatial.total_genes}")
print(f"  → 每个变换增加 4 个 bbox 基因")

# 手动设置一些基因来演示
# 格式: [enable, params..., x1, y1, x2, y2] for each transform
print("\n设置示例基因:")
chromosome_spatial.genes = np.array([
    # 变换1: Rotate
    0.9,           # 启用
    0.5,           # limit 参数
    0.0, 0.0, 0.5, 0.5,  # bbox: 左上角

    # 变换2: HueSaturationValue
    0.8,           # 启用
    0.3, 0.4, 0.2, # 3个参数
    0.5, 0.0, 1.0, 0.5,  # bbox: 右上角

    # 变换3: GaussNoise
    0.7,           # 启用
    0.6, 0.8,      # 2个参数
    0.25, 0.25, 0.75, 0.75,  # bbox: 中心区域
])

summary = chromosome_spatial.get_transform_summary()
print("\n变换序列（带空间定位）:")
for i, info in enumerate(summary, 1):
    print(f"\n  变换 {i}: {info['name']}")
    print(f"    启用: {info['enabled']}")
    print(f"    参数: {info['params']}")
    if 'bbox' in info:
        bbox = info['bbox']
        print(f"    区域: ({bbox[0]:.2f}, {bbox[1]:.2f}) → ({bbox[2]:.2f}, {bbox[3]:.2f})")


# ==================== 示例3: 从传统 transform_sequence 转换 ====================

print("\n\n【示例3】从传统 transform_sequence 转换:")
print("-" * 80)

# 传统方式：使用变换ID列表
transform_sequence = [2, 27, 32]  # ShiftScaleRotate, GaussNoise, Blur

# 转换为新格式
configs = get_configs_from_sequence(transform_sequence, fixed_mode=True)

# 创建染色体
chromosome_legacy = EnhancedChromosome(configs, spatial_enabled=False)

print(f"输入 transform_sequence: {transform_sequence}")
print(f"转换后的变换:")
for i, config in enumerate(configs):
    print(f"  {i+1}. ID={transform_sequence[i]} → {config.name}")

print(f"\n总基因数: {chromosome_legacy.total_genes}")


# ==================== 示例4: 应用变换到图像 ====================

print("\n\n【示例4】应用变换到图像:")
print("-" * 80)

# 创建测试图像
test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
print(f"测试图像: {test_image.shape}")

# 使用平衡型配置
configs = get_balanced_config(fixed_mode=True)
chromosome = EnhancedChromosome(configs, spatial_enabled=False)

# 解码为 Albumentations 变换
albumentations_transform = chromosome.decode()

print(f"\n创建的变换序列: A.Compose({len(albumentations_transform.transforms)} 个变换)")

# 应用变换
try:
    result = albumentations_transform(image=test_image)
    transformed_image = result['image']
    print(f"变换后图像: {transformed_image.shape}")

    # 计算差异
    diff = np.abs(transformed_image.astype(float) - test_image.astype(float)).mean()
    print(f"平均像素差异: {diff:.2f}")

    print("\n✅ 变换应用成功！")
except Exception as e:
    print(f"\n❌ 变换应用失败: {e}")


# ==================== 示例5: 比较不同配置 ====================

print("\n\n【示例5】比较不同预设配置:")
print("-" * 80)

config_funcs = [
    ("轻量型", get_lightweight_config),
    ("平衡型", get_balanced_config),
    ("完整型", get_comprehensive_config),
    ("语义保持型", get_semantic_preserving_config),
    ("对抗型", get_adversarial_config),
]

print(f"\n{'配置名称':<15} {'变换数':<10} {'基因数(基础)':<15} {'基因数(空间)':<15}")
print("-" * 60)

for name, func in config_funcs:
    configs = func(fixed_mode=True)

    # 基础模式基因数
    genes_basic = sum(1 + c.num_params for c in configs)

    # 空间模式基因数
    genes_spatial = sum(1 + c.num_params + 4 for c in configs)

    print(f"{name:<15} {len(configs):<10} {genes_basic:<15} {genes_spatial:<15}")


# ==================== 使用建议 ====================

print("\n\n" + "=" * 80)
print("使用建议")
print("=" * 80)

print("""
1. 快速开始（推荐）:
   ✓ 使用 get_balanced_config(fixed_mode=True)
   ✓ 5个变换，覆盖几何、颜色、噪声、模糊、遮挡
   ✓ 基因数适中（约15-20个）

2. 快速测试:
   ✓ 使用 get_lightweight_config(fixed_mode=True)
   ✓ 只有3个变换，计算快速
   ✓ 适合原型验证、参数调优

3. 全面测试:
   ✓ 使用 get_comprehensive_config(fixed_mode=True)
   ✓ 10个变换，全面覆盖各种类型
   ✓ 适合最终鲁棒性测试

4. 特定场景:
   ✓ 语义分割: get_semantic_preserving_config()
   ✓ 对抗样本: get_adversarial_config()

5. 兼容旧代码:
   ✓ 使用 get_configs_from_sequence([2, 27, 32], fixed_mode=True)
   ✓ 将 transform_sequence 转换为新格式

6. 参数选择:
   ✓ fixed_mode=True:  推荐，确保可复现性，节省基因数
   ✓ fixed_mode=False: 用于数据增强，增加多样性

   ✓ spatial_enabled=False: 推荐，全图变换
   ✓ spatial_enabled=True:  高级用法，区域定位变换
""")

print("\n" + "=" * 80)
print("代码模板")
print("=" * 80)

print("""
# 最简单的使用方式
from mostest.encoding.default_configs import get_balanced_config
from mostest.encoding.chromosome_enhanced import EnhancedChromosome

# 1. 创建配置
configs = get_balanced_config(fixed_mode=True)

# 2. 创建染色体
chromosome = EnhancedChromosome(configs, spatial_enabled=False)

# 3. 应用到图像
transform = chromosome.decode()
result = transform(image=your_image)
transformed_image = result['image']
""")
