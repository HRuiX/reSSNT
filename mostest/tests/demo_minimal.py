"""
最小演示脚本 - 使用模拟模型快速验证框架
Minimal Demo Script - Quick framework verification with mock model
"""

import torch
import torch.nn as nn
import numpy as np
from main import MOSTest


class MockSegmentationModel(nn.Module):
    """
    模拟的语义分割模型
    Mock Semantic Segmentation Model
    """

    def __init__(self, num_classes=150):
        super().__init__()

        # 简单的编码器-解码器结构 Simple encoder-decoder structure
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        """前向传播 Forward pass"""
        features = self.encoder(x)
        output = self.decoder(features)
        return output


def create_mock_data(num_samples=3, height=128, width=128, num_classes=150):
    """
    创建模拟数据
    Create mock data

    Args:
        num_samples: 样本数量
        height, width: 图像尺寸
        num_classes: 类别数量

    Returns:
        leftImg8bit, masks
    """
    images = []
    masks = []

    for _ in range(num_samples):
        # 创建随机图像 Create random image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # 创建随机掩码 Create random mask
        mask = np.random.randint(0, num_classes, (height, width), dtype=np.uint8)

        images.append(image)
        masks.append(mask)

    return images, masks


def main():
    """主函数 Main function"""

    print("="*80)
    print("MOSTest 最小演示 MOSTest Minimal Demo")
    print("="*80)

    # ========== 1. 创建模拟模型 Create mock model ==========
    print("\n步骤1: 创建模拟模型 Step 1: Creating mock model...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备 Using device: {device}")

    model = MockSegmentationModel(num_classes=150)
    model = model.to(device)
    model.eval()

    print("✓ 模型创建成功 Model created successfully")

    # ========== 2. 创建模拟数据 Create mock data ==========
    print("\n步骤2: 创建模拟数据 Step 2: Creating mock data...")

    seed_images, seed_masks = create_mock_data(
        num_samples=3,
        height=128,
        width=128,
        num_classes=150
    )

    print(f"✓ 创建了 {len(seed_images)} 个种子样本 Created {len(seed_images)} seed samples")
    print(f"  图像形状 Image shape: {seed_images[0].shape}")
    print(f"  掩码形状 Mask shape: {seed_masks[0].shape}")

    # ========== 3. 初始化MOSTest ==========
    print("\n步骤3: 初始化MOSTest Step 3: Initializing MOSTest...")

    mostest = MOSTest(
        model=model,
        device=device,
        population_size=20,      # 小种群用于快速演示 Small population for quick demo
        max_generations=10,      # 少迭代次数用于快速演示 Few generations for quick demo
        num_classes=150,
        output_dir='./mostest_demo_output'
    )

    print("✓ MOSTest初始化成功 MOSTest initialized successfully")

    # ========== 4. 运行优化 Run optimization ==========
    print("\n步骤4: 运行优化 Step 4: Running optimization...")
    print("(这可能需要几分钟...) (This may take a few minutes...)")

    results = mostest.run(
        seed_images=seed_images,
        seed_masks=seed_masks,
        verbose=True
    )

    # ========== 5. 查看结果 View results ==========
    print("\n" + "="*80)
    print("结果摘要 Results Summary")
    print("="*80)

    print(f"\n生成的测试样本数 Generated test samples: {len(results['test_samples'])}")

    print(f"\n最终覆盖率 Final coverage:")
    print(f"  TKNP (Top-K神经元模式): {results['final_coverage']['tknp']:.3f}")
    print(f"  SBC  (语义边界):        {results['final_coverage']['sbc']:.3f}")
    print(f"  ADC  (激活分布):        {results['final_coverage']['adc']:.3f}")

    print(f"\nPareto前沿样本目标函数值 Pareto front sample objectives:")
    for i, sample in enumerate(results['test_samples'][:5]):
        f1, f2, f3 = sample['objectives']
        print(f"  样本 Sample {i}: F1={f1:.4f}, F2={f2:.4f}, F3={f3:.4f}")

    print(f"\n覆盖率演化 Coverage evolution:")
    tknp_history = results['history']['coverage_history']['tknp']
    sbc_history = results['history']['coverage_history']['boundary']
    adc_history = results['history']['coverage_history']['activation']

    print(f"  代数 Gen | TKNP  | SBC   | ADC")
    print(f"  " + "-"*35)
    for gen in range(0, len(tknp_history), max(1, len(tknp_history)//5)):
        print(f"  {gen:4d}    | {tknp_history[gen]:.3f} | {sbc_history[gen]:.3f} | {adc_history[gen]:.3f}")

    # ========== 6. 清理资源 Cleanup ==========
    print("\n步骤5: 清理资源 Step 5: Cleaning up...")
    mostest.cleanup()

    print("\n" + "="*80)
    print("✓ 演示完成！Demo completed!")
    print(f"结果已保存到 Results saved to: {mostest.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
