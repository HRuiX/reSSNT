"""
MOSTest 配置文件
MOSTest Configuration File

集中管理所有超参数，包括随机种子、种群参数、优化参数等
Centralized management of all hyperparameters including random seed, population parameters, optimization parameters, etc.
"""

import random
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path


class MOSTestConfig:
    """
    MOSTest 配置类
    MOSTest Configuration Class

    使用方法：
    1. 直接使用预定义配置: config = MOSTestConfig.small_scale()
    2. 自定义配置: config = MOSTestConfig(population_size=100, ...)
    3. 从字典创建: config = MOSTestConfig.from_dict(my_config_dict)
    4. 从文件加载: config = MOSTestConfig.from_file('config.yaml')
    """

    def __init__(
        self,
        # ==================== 模型配置 Model Configuration ====================
        model_name: str ,
        dataset: str ,
        model_type: str,
        config_file: str ,
        checkpoint_file: str,
        device: str,
        num_classes: int,

        # ==================== 随机种子 Random Seed ====================
        random_seed: Optional[int] = 42,

        # ==================== 并行优化参数 Parallel Optimization Parameters ====================
        num_workers: Optional[int] = 30,  # 并行工作进程数（None=自动检测）

        # ==================== 种群参数 Population Parameters ====================
        population_size: int = 100,
        max_generations: int = 1000,
        max_runtime_hours: int = 12,
        psnr_threshold:int = 15,

        # ==================== 覆盖率参数 Coverage Parameters ====================
        top_k: int = 25,              # TKNP覆盖率的top-k参数
        num_bins: int = 100,          # ADC覆盖率的bins数量

        # ==================== NSGA-III 参数 NSGA-III Parameters ====================
        num_objectives: int = 3,       # 目标函数数量
        # ref_point_divisions: int = 15,  # 参考点分层参数
        ref_point_divisions: int = 12,  # 参考点分层参数
        crossover_prob: float = 0.9,   # 交叉概率
        tournament_size: int = 2,      # 锦标赛选择大小

        # ==================== 遗传算子参数 Genetic Operator Parameters ====================
        crossover_eta: float = 30.0,   # SBX交叉分布指数

        # ==================== 停止准则参数 Stopping Criteria Parameters ====================
        tknp_threshold: float = 0.20,   # TKNP覆盖率阈值
        sbc_threshold: float = 0.90,    # SBC边界覆盖率阈值
        adc_threshold: float = 0.95,    # ADC激活覆盖率阈值
        stagnation_generations: int = 20,     # 停滞检测代数
        stagnation_tolerance: float = 0.001,  # 停滞容忍度

        # ==================== 输出配置 Output Configuration ====================
        output_dir: str = None,
        save_generation_interval: int = 1,    # 每N代保存一次完整数据
        save_images: bool = True,             # 是否保存图像
        verbose: bool = True,                 # 是否显示详细信息

        # ==================== 变换序列配置 Transform Sequence Configuration ====================
        transform_sequence: list = None,  # balanced, aggressive, conservative
        spatial_enabled: bool = True,            # 是否启用空间变换
        single_transform_init: bool = True,       # 初始种群是否单变换

        # 染色体编码参数 Chromosome Encoding Parameters
        enable_threshold: float = 0.5,            # 变换启用阈值
        modify_pic_cnt: int = 50,                 # 修改指示图片
        disabled_gene_range: tuple = (0.3, 0.45), # 禁用基因范围
        enabled_gene_range: tuple = (0.7, 1.0),   # 启用基因范围
    ):
        """
        初始化配置

        Args:
            所有参数见上方注释
        """
        # 随机种子
        self.random_seed = random_seed

        # 模型配置
        self.model_name = model_name
        self.dataset = dataset
        self.model_type = model_type
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.num_classes = num_classes

        # 种群参数
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_runtime_hours = max_runtime_hours
        self.psnr_threshold = psnr_threshold

        # 覆盖率参数
        self.top_k = top_k
        self.num_bins = num_bins

        # NSGA-III参数
        self.num_objectives = num_objectives
        self.ref_point_divisions = ref_point_divisions
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size

        # 遗传算子参数
        self.crossover_eta = crossover_eta

        # 并行优化参数
        self.num_workers = num_workers

        # 停止准则参数
        self.tknp_threshold = tknp_threshold
        self.sbc_threshold = sbc_threshold
        self.adc_threshold = adc_threshold
        self.stagnation_generations = stagnation_generations
        self.stagnation_tolerance = stagnation_tolerance

        # 输出配置
        self.output_dir = output_dir
        self.save_generation_interval = save_generation_interval
        self.save_images = save_images
        self.verbose = verbose

        # 变换序列配置
        self.transform_sequence = transform_sequence
        self.spatial_enabled = spatial_enabled
        self.single_transform_init = single_transform_init

        # 染色体编码参数
        self.enable_threshold = enable_threshold
        self.modify_pic_cnt = modify_pic_cnt
        self.disabled_gene_range = disabled_gene_range
        self.enabled_gene_range = enabled_gene_range

    def set_random_seed(self):
        """
        设置所有随机种子以确保可重复性
        Set all random seeds for reproducibility
        """
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_seed)
                torch.cuda.manual_seed_all(self.random_seed)

            print(f"✓ 随机种子已设置: {self.random_seed}")

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        Convert to dictionary
        """
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MOSTestConfig':
        """
        从字典创建配置
        Create configuration from dictionary
        """
        return cls(**config_dict)

    def save(self, filepath: str):
        """
        保存配置到文件
        Save configuration to file
        """
        import json
        filepath = Path(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"✓ 配置已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MOSTestConfig':
        """
        从文件加载配置
        Load configuration from file
        """
        import json
        filepath = Path(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        print(f"✓ 配置已从文件加载: {filepath}")
        return cls.from_dict(config_dict)

    def print_summary(self):
        """
        打印配置摘要
        Print configuration summary
        """
        print("=" * 80)
        print("MOSTest 配置摘要 / Configuration Summary")
        print("=" * 80)

        print("\n【模型配置】")
        print(f"  模型: {self.model_name} ({self.model_type})")
        print(f"  数据集: {self.dataset}")
        print(f"  类别数: {self.num_classes}")

        print("\n【种群参数】")
        print(f"  种群大小: {self.population_size}")
        print(f"  最大代数: {self.max_generations}")
        print(f"  最大运行时间: {self.max_runtime_hours}小时")

        print("\n【并行优化】")
        if self.parallel:
            mode = "多进程" if self.use_multiprocessing else "多线程"
            print(f"  并行模式: {mode}")
            print(f"  工作进程数: {self.num_workers if self.num_workers else '自动'}")
            print(f"  批量推理: {'是' if self.batch_inference else '否'} (batch_size={self.batch_size})")
            print(f"  异步I/O: {'是' if self.async_io else '否'}")
        else:
            print(f"  并行模式: 串行")

        print("\n【其他配置】")
        print(f"  随机种子: {self.random_seed}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  详细输出: {'是' if self.verbose else '否'}")

        print("=" * 80)

    # ========================================================================
    # 预定义配置模板
    # Predefined Configuration Templates
    # ========================================================================

    @classmethod
    def small_scale(cls, **kwargs) -> 'MOSTestConfig':
        """
        小规模测试配置
        Small-scale test configuration

        适用场景：
        - 快速测试和调试
        - 资源受限环境
        - 种群<30
        """
        config = cls(
            population_size=20,
            max_generations=50,
            max_runtime_hours=1,
            parallel=False,  # 小规模不需要并行
            batch_inference=False,
            **kwargs
        )
        return config

    @classmethod
    def medium_scale(cls, **kwargs) -> 'MOSTestConfig':
        """
        中等规模配置（推荐）
        Medium-scale configuration (Recommended)

        适用场景：
        - 常规实验
        - 种群30-100
        - 平衡性能和效果
        """
        config = cls(
            population_size=50,
            max_generations=100,
            max_runtime_hours=6,
            parallel=True,
            use_multiprocessing=False,  # 使用多线程，避免pickle错误
            num_workers=4,
            batch_inference=True,
            batch_size=8,
            async_io=True,
            **kwargs
        )
        return config

    @classmethod
    def large_scale(cls, **kwargs) -> 'MOSTestConfig':
        """
        大规模配置
        Large-scale configuration

        适用场景：
        - 大规模实验
        - 种群>100
        - 充足的计算资源
        """
        config = cls(
            population_size=100,
            max_generations=200,
            max_runtime_hours=12,
            parallel=True,
            use_multiprocessing=False,  # 使用多线程，避免pickle错误
            num_workers=8,
            batch_inference=True,
            batch_size=16,
            async_io=True,
            **kwargs
        )
        return config

    @classmethod
    def debug_config(cls, **kwargs) -> 'MOSTestConfig':
        """
        调试配置
        Debug configuration

        适用场景：
        - 代码调试
        - 功能验证
        - 快速迭代
        """
        config = cls(
            population_size=10,
            max_generations=5,
            max_runtime_hours=0.5,
            parallel=False,
            batch_inference=False,
            save_images=False,  # 调试时不保存图像
            verbose=True,
            **kwargs
        )
        return config

    @classmethod
    def low_memory(cls, **kwargs) -> 'MOSTestConfig':
        """
        低内存配置（极限优化）
        Low-memory configuration (Extreme optimization)

        适用场景：
        - GPU显存<8GB
        - 经常遇到 OOM 错误
        - 需要最小内存占用
        """
        config = cls(
            population_size=10,     # 最小种群
            max_generations=30,     # 减少迭代
            max_runtime_hours=3,
            parallel=False,         # 禁用并行
            batch_inference=False,  # 禁用批量推理
            batch_size=1,
            async_io=True,
            top_k=10,              # 减少关注的神经元
            **kwargs
        )
        return config

    @classmethod
    def gpu_limited(cls, **kwargs) -> 'MOSTestConfig':
        """
        GPU受限配置
        GPU-limited configuration

        适用场景：
        - GPU显存<8GB
        - 多人共享GPU
        """
        config = cls(
            population_size=30,
            max_generations=100,
            max_runtime_hours=6,
            parallel=True,
            use_multiprocessing=False,  # 使用多线程，避免pickle错误
            num_workers=2,      # 减少线程数避免GPU竞争
            batch_inference=True,
            batch_size=4,       # 小batch避免OOM
            async_io=True,
            **kwargs
        )
        return config

    @classmethod
    def high_performance(cls, **kwargs) -> 'MOSTestConfig':
        """
        高性能配置（最大化加速）
        High-performance configuration (Maximum acceleration)

        适用场景：
        - 充足的计算资源（8+核CPU，16GB+ GPU）
        - 追求最快速度
        """
        config = cls(
            population_size=100,
            max_generations=100,
            max_runtime_hours=6,
            parallel=True,
            use_multiprocessing=False,  # 使用多线程，避免pickle错误
            num_workers=8,
            batch_inference=True,      # 批量推理
            batch_size=16,
            async_io=True,             # 异步I/O
            **kwargs
        )
        return config

    @classmethod
    def reproducible(cls, **kwargs) -> 'MOSTestConfig':
        """
        可重复配置（固定随机种子）
        Reproducible configuration (Fixed random seed)

        适用场景：
        - 需要可重复的实验结果
        - 对比实验
        """
        config = cls(
            random_seed=42,
            deterministic=True,  # 启用确定性算法
            cudnn_benchmark=False,
            **kwargs
        )
        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置对象转换为字典
        Convert config object to dictionary
        """
        import inspect
        config_dict = {}

        # 获取__init__方法的参数
        init_signature = inspect.signature(self.__init__)
        for param_name in init_signature.parameters:
            if hasattr(self, param_name):
                config_dict[param_name] = getattr(self, param_name)

        return config_dict

    def save_to_file(self, file_path: str):
        """
        保存配置到JSON文件
        Save configuration to JSON file

        Args:
            file_path: 保存路径 (Path to save the configuration)
        """
        import json
        config_dict = self.to_dict()

        # 确保目录存在
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ 配置已保存到: {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str) -> 'MOSTestConfig':
        """
        从JSON文件加载配置
        Load configuration from JSON file

        Args:
            file_path: 配置文件路径 (Path to the configuration file)

        Returns:
            MOSTestConfig: 配置对象 (Configuration object)
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        print(f"✓ 配置已从文件加载: {file_path}")
        return cls(**config_dict)

    def print_summary(self):
        """
        打印配置摘要
        Print configuration summary
        """
        print("=" * 80)
        print("MOSTestConfig Summary")
        print("=" * 80)

        print("\n[Model Configuration]")
        print(f"  model_name         : {self.model_name}")
        print(f"  dataset            : {self.dataset}")
        print(f"  model_type         : {self.model_type}")
        print(f"  config_file        : {self.config_file}")
        print(f"  checkpoint_file    : {self.checkpoint_file}")
        print(f"  device             : {self.device}")
        print(f"  num_classes        : {self.num_classes}")

        print("\n[Random Seed]")
        print(f"  random_seed        : {self.random_seed}")
        print(f"  deterministic      : {self.deterministic}")
        print(f"  cudnn_benchmark    : {self.cudnn_benchmark}")

        print("\n[Population Parameters]")
        print(f"  population_size    : {self.population_size}")
        print(f"  max_generations    : {self.max_generations}")
        print(f"  max_runtime_hours  : {self.max_runtime_hours}")

        print("\n[Parallel Optimization]")
        print(f"  parallel           : {self.parallel}")
        print(f"  use_multiprocessing: {self.use_multiprocessing}")
        print(f"  num_workers        : {self.num_workers}")
        print(f"  batch_inference    : {self.batch_inference}")
        print(f"  batch_size         : {self.batch_size}")
        print(f"  async_io           : {self.async_io}")

        print("\n[Coverage Parameters]")
        print(f"  top_k              : {self.top_k}")
        print(f"  num_bins           : {self.num_bins}")

        print("\n[NSGA-III Parameters]")
        print(f"  num_objectives     : {self.num_objectives}")
        print(f"  ref_point_divisions: {self.ref_point_divisions}")
        print(f"  crossover_prob     : {self.crossover_prob}")
        print(f"  tournament_size    : {self.tournament_size}")

        print("\n[Output Configuration]")
        print(f"  output_dir         : {self.output_dir}")
        print(f"  save_images        : {self.save_images}")
        print(f"  verbose            : {self.verbose}")

        print("=" * 80)



# ==================== 辅助函数 Helper Functions ====================

def get_config(config_type: str = "medium", **kwargs):
    """
    获取预定义配置
    Get predefined configuration

    Args:
        config_type: 配置类型 (Configuration type)
            - "small": 小规模测试配置
            - "medium": 中等规模配置 (推荐)
            - "large": 大规模实验配置
            - "debug": 调试配置
            - "gpu_limited": GPU受限配置
            - "high_performance": 高性能配置
            - "reproducible": 可重复配置
        **kwargs: 额外参数用于覆盖默认配置

    Returns:
        MOSTestConfig: 配置对象

    Example:
        >>> config = get_config("medium")
        >>> config.model_name = "segformer"
        >>> config.dataset = "ade20k"
    """
    config_map = {
        "small": MOSTestConfig.small_scale,
        "medium": MOSTestConfig.medium_scale,
        "large": MOSTestConfig.large_scale,
        "debug": MOSTestConfig.debug_config,
        "gpu_limited": MOSTestConfig.gpu_limited,
        "high_performance": MOSTestConfig.high_performance,
        "reproducible": MOSTestConfig.reproducible,
    }

    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}. "
                         f"Available types: {list(config_map.keys())}")

    return config_map[config_type](**kwargs)


def create_custom_config(base_config: str = "medium", **kwargs):
    """
    创建自定义配置（基于预定义配置）
    Create custom configuration (based on predefined config)

    Args:
        base_config: 基础配置类型 (Base configuration type)
        **kwargs: 要覆盖的参数 (Parameters to override)

    Returns:
        MOSTestConfig: 自定义配置对象

    Example:
        >>> config = create_custom_config(
        ...     base_config="medium",
        ...     model_name="segformer",
        ...     dataset="ade20k",
        ...     population_size=80
        ... )
    """
    return get_config(base_config, **kwargs)
