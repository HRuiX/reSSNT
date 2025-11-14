import torch
from mmengine.runner import Runner
from mmengine.config import Config
import coverage
import os
import os.path as osp
from mmengine.dataset import Compose
from hook import OutputSizeHook, SaveLayerInputOutputHook, CoverageAddHook
import utility
from functools import partial
import cal_cov
from torch.utils.data import DataLoader
import analyse
import perpa_data
from rich import print
import cal_diversity
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch.nn as nn


class CoverageTest:
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_SEED = 42
    DEFAULT_CHOICE_SAMPLES = 100

    def __init__(
            self,
            model_name: str,
            dataset: str,
            model_type: str,
            config: str,
            checkpoint: str,
            dataset_path_prefix: str,
            mode: str = "test",
            choice_samples: int = DEFAULT_CHOICE_SAMPLES,
            coverages_setting: Optional[Dict] = None,
            fuzz: bool = False,
            cov_select_type: str = None,
            save_data_path: str = None,
    ):
        """
        初始化覆盖率测试类

        Args:
            model_name: 模型名称
            dataset: 数据集名称
            model_type: 模型类型
            config: 配置文件路径
            checkpoint: 检查点文件路径
            dataset_path_prefix: 数据集路径前缀
            mode: 运行模式 ("test", "train", "choice_test")
            choice_samples: 选择样本数量
            coverages_setting: 覆盖率设置
            fuzz: 是否使用模糊测试
        """
        # 模型设置
        self.model_name = model_name
        self.dataset = dataset
        self.config = config
        self.checkpoint = checkpoint
        self.device = self._get_device()
        self.mode = mode
        self.model_type = model_type
        self.fuzz = fuzz

        # 构建运行器和管道
        self.runner, self.pipeline = self._build_runner(config)
        self.transforms = Compose(self.pipeline)
        self.model = self.runner.model
        self.model.eval()

        # 路径设置
        self.dataset_path_prefix = dataset_path_prefix
        self._setup_paths(save_data_path)

        # 获取输出尺寸和层名称
        self.layer_dict = utility.get_model_layers(self)
        self.output_sizes, self.layer_dict_keys = self._get_output_size()

        # 覆盖率设置
        self.coverages_setting = self._build_coverage_setting(coverages_setting)
        print(f"覆盖率设置: {self.coverages_setting}")

        # Hook预定义
        self.save_layer_io_hook = None

        # 其他设置
        self.seed = self.DEFAULT_SEED
        self.choice_samples = choice_samples
        self.num_workers = self.DEFAULT_NUM_WORKERS
        self.batch_size = self.DEFAULT_BATCH_SIZE

        self.cov_select_type = cov_select_type
        
        self_data_recordsss = []

        # 模糊测试设置
        if fuzz:
            self._setup_fuzz_parameters()

    def _get_device(self) -> torch.device:
        """获取设备"""
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _setup_paths(self,save_data_path) -> None:
        """设置路径"""
        if save_data_path is None:
            save_data_path = "None"
        self.data_save_path_prefix = f"./output-data-{save_data_path}/{self.dataset}/{self.model_type}/{self.model_name}"
        self.TOOL_LOG_FILE_PATH = f"./New_temp/{self.dataset}/{self.model_type}/{self.model_name}"
        utility.build_path([self.data_save_path_prefix, self.TOOL_LOG_FILE_PATH])
        self.coverage_save_path = f"{self.data_save_path_prefix}/all_test_cov"
        if self.fuzz:
            self.fuzz_save_path_prefix = f"./output-fuzz-data-{save_data_path}/"


    def _setup_fuzz_parameters(self) -> None:
        """设置模糊测试参数"""
        # More robust way to get the first key
        key = next(iter(self.coverages_setting))
        val = self.coverages_setting[key][0]
        val = str(val).replace(".", "")

        if len(self.coverages_setting) == 1:
            self.fuzz_save_path_prefix = f"./output-fuzz-data-0907/{key}-{val}/{self.dataset}/{self.model_type}/fuzz_test"
        utility.build_path([self.fuzz_save_path_prefix])

        self.mutate_batch_size = 1
        self.num_class = 19 if self.dataset == 'cityscapes' else 150
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        # 模糊测试超参数
        self.alpha = 0.2
        self.beta = 0.5
        self.TRY_NUM = 50

        
        self.coverages = self._build_coverage()

    def _build_coverage_setting(self, coverages_setting: Optional[Dict]) -> Dict:
        """构建覆盖率设置"""
        if coverages_setting is not None:
            return coverages_setting

        # 默认设置
        base_cc_value = [5, 10, 19] if self.dataset == 'cityscapes' else [50, 100, 150]

        if not self.fuzz:
            return {
                "NC": [0.25, 0.5, 0.75],
                "KMNC": [25, 50, 100, 1000],
                'SNAC': [None],
                'NBC': [None],
                'TKNC': [5, 10, 15],
                'TKNP': [10, 25, 50],
                'CC':  base_cc_value,
                'NLC': [None],
            }
        else:
            nc_values = [0.5, 0.75] if self.model_type == "CNN" else [0.25, 0.75]

            return {
                "NC": nc_values,
                "KMNC": [100],
                'SNAC': [None],
                'NBC': [None],
                'TKNC': [15],
                'CC': [19] if self.dataset == 'cityscapes' else [50, 150],
                'TKNP': [25],
                'NLC': [None],
            }

    def _build_runner(self, config: str) -> tuple:
        """构建运行器"""
        cfg = Config.fromfile(config)

        if self.mode == "train":
            cfg.val_dataloader.dataset.data_prefix = cfg.train_dataloader.dataset.data_prefix

        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
        cfg.load_from = self.checkpoint
        cfg.launcher = "none"

        runner = Runner.from_cfg(cfg)
        runner.model.cfg = cfg

        # 构建处理管道
        pipeline = self._build_pipeline(cfg.test_pipeline)

        return runner, pipeline

    def _build_pipeline(self, test_pipeline: List) -> List:
        """构建处理管道"""
        pipeline = []
        excluded_types = {"LoadAnnotations", "GenerateEdge"}

        for pip in test_pipeline:
            if (pip['type'] not in excluded_types and
                    "cat_max_ratio" not in pip.keys()):
                pipeline.append(pip)

        return pipeline

    def _get_output_size(self) -> tuple:
        """获取输出尺寸"""
        outputsizehook = OutputSizeHook(self)
        return outputsizehook.output_sizes, outputsizehook.layer_dict_keys

    def _build_coverage(self) -> Dict:
        """构建覆盖率指标"""
        covers = {}
        for key, value in self.coverages_setting.items():
            cov = partial(
                getattr(coverage, key),
                model_name=self.model_name,
                layer_size_dict=self.output_sizes,
                device=self.device,
                save_path=self.coverage_save_path,
                TOOL_LOG_FILE_PATH=self.TOOL_LOG_FILE_PATH
            )

            if isinstance(value, (int, type(None))):
                covers[f"{key}-{value}"] = [cov(threshold=value)]
            elif isinstance(value, list):
                for v in value:
                    covers[f"{key}-{v}"] = [cov(threshold=v)]

        return covers

    def recovery_coverages(self) -> None:
        """恢复覆盖率指标"""
        self.coverages = self._build_coverage()
        for key, covers in tqdm(self.coverages.items(), desc="Recovery Coverage Metrics"):
            cover = covers[0]
            cover.recvoery_progress("test")

        self.print_coverages_info()

    def print_coverages_info(self) -> None:
        re_data = {}
        for key, covers in self.coverages.items():
            cover = covers[0]
            print(f"Current Coverage of {key} is {cover.current}.")
            re_data[key] = cover.current
            
        torch.save(re_data, f"{self.data_save_path_prefix}/ALL-{self.mode}-{Path(self.config).stem}-cov.pth")
        return re_data

    def analyze_hooke_main(self) -> Dict:
        """分析主函数"""
        self.coverages = self._build_coverage()

        save_layer_io_hook = SaveLayerInputOutputHook(self)
        self.runner.register_hook(save_layer_io_hook)
        
        self.print_coverages_info()
        metrics = self.runner.val()
        name = Path(self.config).stem

        return {name: metrics}

    def _process_build_coverage(self, exec_type: str) -> None:
        """处理构建覆盖率"""
        self._rebuild_model(self.config)

        self.coverage_save_path = f"{self.data_save_path_prefix}/all_{exec_type}_cov"
        utility.build_path([self.coverage_save_path])

        res = self.analyze_hooke_main()
        config_name = Path(self.config).stem

        print(utility.dict_to_table(res[config_name]))
        data = [config_name,res[config_name]['mIoU'],res[config_name]['mAcc'],res[config_name]['aAcc']]
        torch.save(data, f"{self.data_save_path_prefix}/ALL-{exec_type}-{config_name}-acc-iou.pth")

        return {
            "mIoU": res[config_name]['mIoU'],
            "mAcc": res[config_name]['mAcc'],
            "aAcc": res[config_name]['aAcc']
        }

    def _rebuild_model(self, config: str) -> None:
        """重建模型"""
        # 清理旧模型
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'runner'):
            del self.runner

        # 构建新模型
        self.runner, _ = self._build_runner(config)
        self.model = self.runner.model
        self.layer_dict = utility.get_model_layers(self)

    def _process_file_for_select(self, save_path: str) -> Dict:
        """处理选择文件"""
        file_name = Path(self.config).stem
        self.coverage_save_path = f"{save_path}/{file_name}/"
        utility.build_path([self.coverage_save_path])

        # 解析覆盖率和超参数
        parts = file_name.split("-")
        cov, hyper = parts[0], parts[1]
        hyper = None if hyper == "None" else float(hyper)

        self.coverages_setting = {cov: [hyper]}
        self._rebuild_model(self.config)

        return self.analyze_hooke_main()

    def pre_build_coverage_info(self,coverages_setting=None) -> None:
        """预构建覆盖率信息"""
        self.mode = "train"
        if coverages_setting is not None:
            self.coverages_setting = coverages_setting
        self._process_build_coverage("train")

    def build_val_coverage_info(self) -> None:
        """构建验证覆盖率信息"""
        self.mode = "test"
        re_iou = self._process_build_coverage("test")
        re_data = self.print_coverages_info()
        cal_cov.cal_ALL_cov(
            self.data_save_path_prefix,
            self.model_name,
            self.mode,
            self.coverages_setting
        )
        re_iou.update(re_data)
        return re_iou
        
        
    def build_diverity_coverage_info(self) -> None:
        """构建验证覆盖率信息"""
        self.mode = "diversity"
        self._process_diversity_coverage("diversity")
        self.print_coverages_info()
        df = cal_cov.cal_ALL_cov(
            self.data_save_path_prefix,
            self.model_name,
            self.mode,
            self.coverages_setting
        )
        config_name = Path(self.config).stem
        df.to_csv(f"{self.data_save_path_prefix}/ALL-{self.mode}-{config_name}_cov.csv",index=False)
        
        
        
    def _process_diversity_coverage(self, exec_type: str) -> None:
        """处理构建覆盖率"""
        self._rebuild_model(self.config)

        self.coverage_save_path = f"{self.data_save_path_prefix}/all_{exec_type}_cov"
        utility.build_path([self.coverage_save_path])

        res = self.analyze_hooke_main()
        config_name = Path(self.config).stem

        print(utility.dict_to_table(res[config_name]))
        torch.save(res, f"{self.data_save_path_prefix}/ALL-{exec_type}-{config_name}-acc-iou.pth")
        

    def build_coverage_info_for_fuzz(self, config: str) -> None:
        self.mode = "fuzz_test_files"
        self.config = config
        self._process_build_coverage("fuzz_test_files")
        self.print_coverages_info()

    def select_files(self, func: Optional[callable] = None) -> None:
        """选择文件"""
        save_path = f"{self.data_save_path_prefix}/all_test_cov"

        self.seclect_file_path = perpa_data.perpa_data_samples(
            save_path,
            self.model_name,
            self.config,
            self.mode,
            self.choice_samples,
            self.dataset,
            self.dataset_path_prefix,
            self.coverages_setting,
            func
        )

    def build_select_coverage_info(self) -> None:
        """构建选择覆盖率信息"""
        self.mode = "choice_test"
        self.seclect_file_path = f"{self.dataset_path_prefix}/diversity/{self.model_name}/config0515"
        files = utility.get_files(self.seclect_file_path)
        save_path = f"{self.data_save_path_prefix}/select_cov"
        if self.cov_select_type is not None:
            save_path = f"{save_path}/add_select"
            
        utility.build_path([save_path])

        self.ALL_FILE_NUM = len(files)

        results = []
        check_path = f"{self.data_save_path_prefix}/Select-acc-iou_Original.pth"
        print(check_path)
        if os.path.exists(check_path):
            print(f"File {check_path} already exists, skipping coverage calculation.")
        else:
            for idx in tqdm(range(len(files)), desc="Executing collection of coverage metrics for various datasets"):
                self.config = files[idx]
                results.append(self._process_file_for_select(save_path))

        # 分析结果
        analyse.analysis_iou(results, self.data_save_path_prefix, self.model_name, self.coverages_setting,cov_select_type = self.cov_select_type)
        self.coverages_setting = self._build_coverage_setting(None)
        cal_cov.cal_select_cov(
            self.data_save_path_prefix,
            # self.data_save_path_prefix,
            self.coverages_setting,
            self.model_name,
            self.mode
        )
      
    def cal_diversity_info(self) -> None:
        self.coverages = self._build_coverage()
        """计算多样性信息"""
        cal_diversity.cal_diversity(
            self.dataset,
            self.dataset_path_prefix,
            self.model_name,
            self.coverages_setting,
            len(self.coverages),
            self.device,
            self.data_save_path_prefix
        )
       
    def analyse_file(self) -> None:
        """分析文件"""
        analyse.analyse_file(self.model_name, self.data_save_path_prefix)

    def cleanup(self) -> None:
        """清理资源"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'runner'):
            del self.runner
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def remove_files(self) -> None:
        """删除文件"""
        import shutil
        shutil.rmtree(self.fuzz_save_path_prefix)

    
    def get_add_cover_info(self) -> None:
        self._rebuild_model(self.config)
        save_path = f"{self.data_save_path_prefix}/all_test_cov"

        flag = False
        path = f"{self.dataset_path_prefix}/diversity/{self.model_name}/config0515/"
        if os.path.exists(path):
           print("No need to add coverage information, all files already exist.")
           self.seclect_file_path = path
        else:
            self.coverages = self._build_coverage()
                
            coverageaddhook = CoverageAddHook(self)
            files_all = coverageaddhook()
           
            self.seclect_file_path = perpa_data.perpa_data_samples(
                save_path,
                self.model_name,
                self.config,
                self.mode,
                self.choice_samples,
                self.dataset,
                self.dataset_path_prefix,
                self.coverages_setting,
                files_path=files_all,
                method="add"
            )
            
            
            