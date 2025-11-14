"""
MOSTest 使用示例 - 使用配置系统
MOSTest Usage Example - Using Configuration System

展示如何使用新的配置系统来运行MOSTest

Version: 4.0 - 中心化配置系统
"""
import sys
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anyio.lowlevel import checkpoint
from mmseg.utils import register_all_modules
from utility1 import get_file_device
from img_conv import get_image_data
from mostest.miane import MOSTest
from mostest.mostconfig import get_config
from mostconfig import MOSTestConfig
import torch
import gc
import warnings
warnings.filterwarnings('ignore')

def main():
    """主函数 Main function"""

    # ========== 1. 注册所有模块 Register all modules ==========
    register_all_modules()

    # ========== 2. 加载模型 Load model ==========
    print("加载模型 Loading model...")
    dataset = "cityscapes"
    model_type = "Transformer"
    model_infos, dataset_path_prefix = get_file_device(dataset, model_type)

    for model_info in model_infos:
        num_classes = 150 if dataset == "ade20k" else 19
        model_name, config_path, checkpoint = model_info[0], model_info[1], model_info[2]
        mostestconfig = MOSTestConfig(
            model_name=model_name,
            dataset=dataset,
            model_type=model_type,
            config_file=config_path,
            checkpoint_file=checkpoint,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_classes=num_classes,
        )

        # ========== 3. 加载种子数据 Load seed data ==========
        print("加载种子数据 Loading seed data...")
        datalist = get_image_data(dataset, dataset_path_prefix)
        gc.collect()
        torch.cuda.empty_cache()

        # ========== 5. 初始化MOSTest ==========
        print("\n初始化MOSTest Initializing MOSTest...")

        # 使用配置对象创建MOSTest实例
        mostest = MOSTest(mostest_config=mostestconfig)

        # ========== 5. 运行优化 Run optimization ==========
        print("\n开始优化 Starting optimization...")

        results = mostest.run(
            datalist=datalist,
            verbose=True
        )

        # ========== 6. 查看结果 View results ==========
        print("\n结果统计 Result statistics:")
        print(f"生成的测试样本数 Generated test samples: {len(results['test_samples'])}")
        print(f"最终覆盖率 Final coverage:")
        print(f"  TKNP: {results['final_coverage']['tknp']:.3f}")
        print(f"  SBC: {results['final_coverage']['sbc']:.3f}")
        print(f"  ADC: {results['final_coverage']['adc']:.3f}")

        # 打印部分Pareto前沿样本的目标函数值
        # Print objective values of some Pareto front samples
        print("\nPareto前沿样本目标函数值 Pareto front sample objectives:")
        for i, sample in enumerate(results['test_samples'][:5]):
            f1, f2, f3 = sample['objectives']
            print(f"  Sample {i}: F1={f1:.4f}, F2={f2:.4f}, F3={f3:.4f}")

        # ========== 7. 清理资源 Cleanup ==========
        mostest.cleanup()

        print("\n完成! Done!")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
