"""
重构的模糊测试运行脚本
参考 NeuraL-Coverage fuzz.py 的执行流程，结合本地项目特性进行重构
"""

import sys
import os
import signal
import argparse
import logging
import warnings
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from rich.text import Text
from rich.rule import Rule
from rich.console import Console
from rich.table import Table

import utility
from fuzz import Fuzzer
from img_conv import build_img_np_list
from config import CoverageTest


# ============================================================================
# 全局配置和初始化
# ============================================================================

console = Console()
logging.disable(logging.CRITICAL)
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(profile="default")


# ============================================================================
# 参数配置类（参考 NeuraL-Coverage 的 Parameters 类）
# ============================================================================

@dataclass
class FuzzParameters:
    """
    模糊测试参数配置类
    参考 NeuraL-Coverage 的 Parameters 设计，整合本地项目的所有配置
    """
    # 模型和数据集配置
    model_name: str = ""
    config: str = ""
    checkpoint: str = ""
    dataset: str = "cityscapes"
    model_type: str = "CNN"
    dataset_path_prefix: str = ""

    # 设备配置
    device: torch.device = field(default_factory=lambda: torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    num_workers: int = 4

    # 批处理配置
    batch_size: int = 50
    mutate_batch_size: int = 1

    # 图像配置
    num_class: int = 19  # cityscapes: 19, ade20k: 150

    # 覆盖率配置
    coverages_setting: Dict = field(default_factory=dict)

    # 模糊测试超参数（参考 NeuraL-Coverage）
    alpha: float = 0.2  # 变异强度参数
    beta: float = 0.5   # 变异边界参数
    TRY_NUM: int = 50   # 变异尝试次数
    K: int = 64         # 功率调度参数
    batch1: int = 64
    batch2: int = 16

    # 优先级调度参数
    p_min: float = 0.01
    gamma: int = 5

    # 运行控制参数
    max_epochs: int = 100000
    max_runtime_hours: int = 6
    save_every: int = 100

    # 并行配置
    num_processes: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    parallel: bool = True

    # 输出路径配置
    output_dir: str = "./output-fuzz-data-0917"
    log_dir: str = "./fuzzer_logs"
    temp_dir: str = "./temp"

    def __post_init__(self):
        """初始化后处理"""
        # 设置类别数
        if self.dataset == 'cityscapes':
            self.num_class = 19
        elif self.dataset == 'ade20k':
            self.num_class = 150

        # 设置默认覆盖率配置
        if not self.coverages_setting:
            self.coverages_setting = self._get_default_coverage_setting()

    def _get_default_coverage_setting(self) -> Dict:
        """获取默认覆盖率设置"""
        cc_value = 19 if self.dataset == 'cityscapes' else 150
        return {
            "NC": [0.75],
            "KMNC": [100],
            'SNAC': [None],
            'NBC': [None],
            'TKNC': [15],
            'CC': [cc_value],
            'TKNP': [25],
            'NLC': [None],
        }

    def update_from_args(self, model_info: Tuple, dataset: str, model_type: str,
                        dataset_path_prefix: str):
        """从模型信息更新参数"""
        self.model_name = model_info[0]
        self.config = model_info[1]
        self.checkpoint = model_info[2]
        self.dataset = dataset
        self.model_type = model_type
        self.dataset_path_prefix = dataset_path_prefix

        # 更新类别数
        self.num_class = 19 if dataset == 'cityscapes' else 150

    def print_summary(self):
        """打印参数摘要"""
        table = Table(title="Fuzzing Parameters", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan", width=30)
        table.add_column("Value", style="green")

        table.add_row("Model Name", self.model_name)
        table.add_row("Dataset", self.dataset)
        table.add_row("Model Type", self.model_type)
        table.add_row("Device", str(self.device))
        table.add_row("Num Classes", str(self.num_class))
        table.add_row("Alpha", str(self.alpha))
        table.add_row("Beta", str(self.beta))
        table.add_row("TRY_NUM", str(self.TRY_NUM))
        table.add_row("K", str(self.K))
        table.add_row("Max Epochs", str(self.max_epochs))
        table.add_row("Max Runtime (hours)", str(self.max_runtime_hours))
        table.add_row("Num Processes", str(self.num_processes))
        table.add_row("Parallel", str(self.parallel))

        console.print(table)


# ============================================================================
# 模糊测试引擎类（参考 NeuraL-Coverage 的 Fuzzer 类结构）
# ============================================================================

class FuzzingEngine:
    """
    模糊测试引擎主类
    参考 NeuraL-Coverage 的 Fuzzer 设计，整合本地项目的 Fuzzer 功能
    """

    def __init__(self, params: FuzzParameters):
        """
        初始化模糊测试引擎

        Args:
            params: 模糊测试参数配置
        """
        self.params = params
        self.coverage_test = None
        self.fuzzer = None
        self.seed_data = None

        # 状态标志
        self.is_initialized = False
        self.is_running = False

        console.print("[bold cyan]初始化模糊测试引擎...[/bold cyan]")
        self._print_banner()

    def _print_banner(self):
        """打印启动横幅"""
        console.print(Rule(style="cyan"))
        console.print(Text("Fuzzing Test Engine (Refactored)", style="bold bright_cyan", justify="center"))
        console.print(Text("Based on NeuraL-Coverage & Local Implementation", style="dim cyan", justify="center"))
        console.print(Rule(style="cyan"))

    def initialize(self) -> bool:
        """
        初始化测试环境
        参考 NeuraL-Coverage 的初始化流程

        Returns:
            bool: 初始化是否成功
        """
        try:
            console.print("\n[bold green]Step 1: 初始化覆盖率测试工具[/bold green]")

            # 创建覆盖率测试对象
            self.coverage_test = CoverageTest(
                model_name=self.params.model_name,
                dataset=self.params.dataset,
                model_type=self.params.model_type,
                config=self.params.config,
                checkpoint=self.params.checkpoint,
                dataset_path_prefix=self.params.dataset_path_prefix,
                fuzz=True,
                coverages_setting=self.params.coverages_setting
            )

            # 将参数传递给覆盖率测试对象
            self.coverage_test.alpha = self.params.alpha
            self.coverage_test.beta = self.params.beta
            self.coverage_test.TRY_NUM = self.params.TRY_NUM
            self.coverage_test.K = self.params.K

            console.print("  ✓ 覆盖率测试工具初始化完成")

            console.print("\n[bold green]Step 2: 构建验证集覆盖率基线[/bold green]")
            self.coverage_test.build_val_coverage_info()
            console.print("  ✓ 验证集覆盖率基线构建完成")

            console.print("\n[bold green]Step 3: 准备种子数据集[/bold green]")
            self.seed_data = build_img_np_list(self.params.config)
            console.print(f"  ✓ 种子数据集准备完成: {len(self.seed_data)} 个种子")

            console.print("\n[bold green]Step 4: 初始化模糊测试器[/bold green]")
            self.fuzzer = Fuzzer(self.coverage_test)
            console.print("  ✓ 模糊测试器初始化完成")

            self.is_initialized = True
            return True

        except Exception as e:
            console.print(f"[bold red]初始化失败: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            return False

    def run(self) -> bool:
        """
        运行模糊测试
        参考 NeuraL-Coverage 的 run 方法流程

        Returns:
            bool: 运行是否成功
        """
        if not self.is_initialized:
            console.print("[bold red]错误: 引擎未初始化[/bold red]")
            return False

        try:
            self.is_running = True

            console.print("\n" + "=" * 80)
            console.print(Text("开始模糊测试", style="bold bright_green", justify="center"))
            console.print("=" * 80 + "\n")

            # 打印参数摘要
            self.params.print_summary()

            console.print("\n[bold cyan]执行模糊测试主循环...[/bold cyan]\n")

            # 调用 Fuzzer.run() - 这里使用本地已有的 Fuzzer 实现
            self.fuzzer.run(self.seed_data)

            console.print("\n[bold green]✓ 模糊测试完成[/bold green]")
            return True

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ 用户中断测试[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[bold red]运行时错误: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.is_running = False

    def exit(self):
        """
        退出并清理
        参考 NeuraL-Coverage 的 exit 方法
        """
        console.print("\n[bold cyan]清理资源...[/bold cyan]")

        if self.fuzzer:
            self.fuzzer.exit()
            console.print("  ✓ 模糊测试器已退出")

        if self.coverage_test:
            self.coverage_test.cleanup()
            console.print("  ✓ 覆盖率测试工具已清理")

        console.print("\n[bold green]✓ 所有资源已清理完毕[/bold green]")
        console.print(Rule(style="green"))


# ============================================================================
# 主程序入口（参考 NeuraL-Coverage 的 main 结构）
# ============================================================================

class FuzzingRunner:
    """
    模糊测试运行器
    参考 NeuraL-Coverage 的主程序结构，提供完整的运行流程
    """

    def __init__(self):
        self.engine = None
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """设置信号处理器（参考 NeuraL-Coverage）"""
        def signal_handler(sig, frame):
            console.print('\n[yellow]⚠ 收到中断信号 (Ctrl+C)[/yellow]')
            if self.engine:
                console.print('[cyan]正在保存当前进度...[/cyan]')
                try:
                    self.engine.exit()
                except Exception as e:
                    console.print(f'[red]退出时发生错误: {e}[/red]')
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    def run_batch_experiments(self,
                             datasets: List[str],
                             model_types: List[str],
                             coverage_settings: Optional[Dict] = None):
        """
        批量运行实验
        参考原 fuzz_run.py 的批量测试逻辑

        Args:
            datasets: 数据集列表
            model_types: 模型类型列表
            coverage_settings: 覆盖率设置
        """
        console.print("\n[bold bright_cyan]批量模糊测试实验[/bold bright_cyan]")
        console.print(f"  数据集: {datasets}")
        console.print(f"  模型类型: {model_types}\n")

        total_experiments = len(datasets) * len(model_types)
        current_experiment = 0

        for dataset in datasets:
            for model_type in model_types:
                current_experiment += 1

                console.print(f"\n{'='*80}")
                console.print(f"实验 {current_experiment}/{total_experiments}")
                console.print(f"数据集: {dataset}, 模型类型: {model_type}")
                console.print(f"{'='*80}\n")

                # 获取模型信息
                model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)

                for model_info in model_infos:
                    try:
                        # 创建参数对象
                        params = FuzzParameters()
                        params.update_from_args(model_info, dataset, model_type, dataset_path_prefix)

                        if coverage_settings:
                            params.coverages_setting = coverage_settings

                        # 创建并运行引擎
                        self.engine = FuzzingEngine(params)

                        if self.engine.initialize():
                            success = self.engine.run()
                            self.engine.exit()

                            if success:
                                console.print(f"[green]✓ 实验完成: {params.model_name}[/green]")
                            else:
                                console.print(f"[yellow]⚠ 实验未成功完成: {params.model_name}[/yellow]")
                        else:
                            console.print(f"[red]✗ 初始化失败: {params.model_name}[/red]")

                    except Exception as e:
                        console.print(f"[red]✗ 实验失败: {e}[/red]")
                        import traceback
                        traceback.print_exc()
                        continue

        console.print("\n[bold green]所有实验完成！[/bold green]")

    def run_single_experiment(self,
                             dataset: str,
                             model_type: str,
                             model_idx: int = 0,
                             coverage_settings: Optional[Dict] = None):
        """
        运行单个实验

        Args:
            dataset: 数据集名称
            model_type: 模型类型
            model_idx: 模型索引
            coverage_settings: 覆盖率设置
        """
        console.print(f"\n[bold bright_cyan]单个模糊测试实验[/bold bright_cyan]")
        console.print(f"  数据集: {dataset}")
        console.print(f"  模型类型: {model_type}\n")

        # 获取模型信息
        model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)

        if model_idx >= len(model_infos):
            console.print(f"[red]错误: 模型索引 {model_idx} 超出范围[/red]")
            return

        model_info = model_infos[model_idx]

        # 创建参数对象
        params = FuzzParameters()
        params.update_from_args(model_info, dataset, model_type, dataset_path_prefix)

        if coverage_settings:
            params.coverages_setting = coverage_settings

        # 创建并运行引擎
        self.engine = FuzzingEngine(params)

        if self.engine.initialize():
            success = self.engine.run()
            self.engine.exit()

            if success:
                console.print(f"\n[green]✓ 实验成功完成[/green]")
            else:
                console.print(f"\n[yellow]⚠ 实验未成功完成[/yellow]")
        else:
            console.print(f"\n[red]✗ 初始化失败[/red]")


# ============================================================================
# 命令行接口（参考 NeuraL-Coverage 的 argparse 设计）
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Fuzzing Test for Semantic Segmentation Models (Refactored)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行单个实验
  python fuzz_run_refactored.py --mode single --dataset cityscapes --model-type CNN

  # 运行批量实验
  python fuzz_run_refactored.py --mode batch --datasets cityscapes ade20k --model-types CNN Transformer

  # 使用原始配置（兼容模式）
  python fuzz_run_refactored.py --mode original
        """
    )

    parser.add_argument('--mode', type=str, default='original',
                       choices=['single', 'batch', 'original'],
                       help='运行模式: single(单个实验), batch(批量实验), original(原始配置)')

    parser.add_argument('--dataset', type=str, default='cityscapes',
                       choices=['cityscapes', 'ade20k'],
                       help='数据集名称（单个实验模式）')

    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['ade20k', 'cityscapes'],
                       help='数据集列表（批量实验模式）')

    parser.add_argument('--model-type', type=str, default='CNN',
                       choices=['CNN', 'Transformer', 'Other'],
                       help='模型类型（单个实验模式）')

    parser.add_argument('--model-types', type=str, nargs='+',
                       default=['CNN', 'Transformer', 'Other'],
                       help='模型类型列表（批量实验模式）')

    parser.add_argument('--model-idx', type=int, default=0,
                       help='模型索引（单个实验模式）')

    # 覆盖率设置
    parser.add_argument('--coverage', type=str, default=None,
                       choices=['NC', 'KMNC', 'SNAC', 'NBC', 'TKNC', 'CC', 'TKNP', 'NLC'],
                       help='覆盖率指标（仅使用单个指标）')

    parser.add_argument('--coverage-threshold', type=float, default=None,
                       help='覆盖率阈值')

    # 运行参数
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='变异强度参数 alpha')

    parser.add_argument('--beta', type=float, default=0.5,
                       help='变异边界参数 beta')

    parser.add_argument('--try-num', type=int, default=50,
                       help='变异尝试次数')

    parser.add_argument('--max-epochs', type=int, default=100000,
                       help='最大epoch数')

    parser.add_argument('--max-hours', type=int, default=6,
                       help='最大运行时间（小时）')

    parser.add_argument('--num-processes', type=int, default=None,
                       help='并行进程数')

    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行处理')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建运行器
    runner = FuzzingRunner()

    # 准备覆盖率设置
    coverage_settings = None
    if args.coverage:
        threshold = args.coverage_threshold
        if threshold is None:
            # 默认阈值
            threshold_map = {
                'NC': 0.75, 'KMNC': 100, 'TKNC': 15,
                'TKNP': 25, 'CC': 19, 'SNAC': None,
                'NBC': None, 'NLC': None
            }
            threshold = threshold_map.get(args.coverage, None)
        coverage_settings = {args.coverage: [threshold]}

    # 根据模式运行
    if args.mode == 'single':
        runner.run_single_experiment(
            dataset=args.dataset,
            model_type=args.model_type,
            model_idx=args.model_idx,
            coverage_settings=coverage_settings
        )

    elif args.mode == 'batch':
        runner.run_batch_experiments(
            datasets=args.datasets,
            model_types=args.model_types,
            coverage_settings=coverage_settings
        )

    elif args.mode == 'original':
        # 原始配置模式（兼容旧版本）
        DATASETS = [[], ["ade20k"], ["cityscapes"], ["ade20k", "cityscapes"]]
        MODEL_TYPE = ["", "CNN", "Transformer", "Other"]

        datasets_idx = 3

        console.print("[bold cyan]运行原始配置模式[/bold cyan]")
        console.print(f"  数据集索引: {datasets_idx}")
        console.print(f"  数据集: {DATASETS[datasets_idx]}")
        console.print(f"  模型类型: CNN, Transformer, Other\n")

        for model_type_idx in [1, 2, 3]:
            model_type = MODEL_TYPE[model_type_idx]
            for dataset in DATASETS[datasets_idx]:
                runner.run_batch_experiments(
                    datasets=[dataset],
                    model_types=[model_type],
                    coverage_settings=coverage_settings
                )


if __name__ == '__main__':
    main()