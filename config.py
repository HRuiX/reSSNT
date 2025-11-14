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
import logging
import cal_diversity
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch.nn as nn

# Configure logger
logger = logging.getLogger(__name__)


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
        Initialize coverage test class

        Args:
            model_name: Model name
            dataset: Dataset name
            model_type: Model type
            config: Configuration file path
            checkpoint: Checkpoint file path
            dataset_path_prefix: Dataset path prefix
            mode: Run mode ("test", "train", "choice_test")
            choice_samples: Number of choice samples
            coverages_setting: Coverage settings
            fuzz: Whether to use fuzzing
            cov_select_type: Coverage selection type
            save_data_path: Path to save data
        """
        # Model settings
        self.model_name = model_name
        self.dataset = dataset
        self.config = config
        self.checkpoint = checkpoint
        self.device = self._get_device()
        self.mode = mode
        self.model_type = model_type
        self.fuzz = fuzz

        # Build runner and pipeline
        self.runner, self.pipeline = self._build_runner(config)
        self.transforms = Compose(self.pipeline)
        self.model = self.runner.model
        self.model.eval()

        # Path settings
        self.dataset_path_prefix = dataset_path_prefix
        self._setup_paths(save_data_path)

        # Get output sizes and layer names
        self.layer_dict = utility.get_model_layers(self.model)
        self.output_sizes, self.layer_dict_keys = self._get_output_size()

        # Coverage settings
        self.coverages_setting = self._build_coverage_setting(coverages_setting)
        logger.info(f"Coverage settings: {self.coverages_setting}")

        # Hook predefinition
        self.save_layer_io_hook = None

        # Other settings
        self.seed = self.DEFAULT_SEED
        self.choice_samples = choice_samples
        self.num_workers = self.DEFAULT_NUM_WORKERS
        self.batch_size = self.DEFAULT_BATCH_SIZE

        self.cov_select_type = cov_select_type

        # Fuzzing settings
        if fuzz:
            self._setup_fuzz_parameters()

        self.print_times = 0

    def _get_device(self) -> torch.device:
        """Get device"""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _setup_paths(self, save_data_path) -> None:
        """Setup paths"""
        if save_data_path is None:
            save_data_path = "None"
        self.data_save_path_prefix = f"./output-data-{save_data_path}/{self.dataset}/{self.model_type}/{self.model_name}"
        self.TOOL_LOG_FILE_PATH = (f"./New_temp/{self.dataset}/{self.model_type}/{self.model_name}")
        utility.build_path([self.data_save_path_prefix, self.TOOL_LOG_FILE_PATH])
        self.coverage_save_path = f"{self.data_save_path_prefix}/all_test_cov"
        if self.fuzz:
            self.fuzz_save_path_prefix = f"./output-fuzz-data-{save_data_path}/"

    def _setup_fuzz_parameters(self) -> None:
        """Setup fuzzing parameters"""
        # More robust way to get the first key
        key = next(iter(self.coverages_setting))
        val = self.coverages_setting[key][0]
        val = str(val).replace(".", "")

        if len(self.coverages_setting) == 1:
            self.fuzz_save_path_prefix = f"{self.fuzz_save_path_prefix}/{key}-{val}/{self.dataset}/{self.model_type}/fuzz_test"
        utility.build_path([self.fuzz_save_path_prefix])

        self.mutate_batch_size = 1
        self.num_class = 19 if self.dataset == "cityscapes" else 150
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        # Fuzzing hyperparameters
        self.alpha = 0.2
        self.beta = 0.5
        self.TRY_NUM = 50

        self.coverages = self._build_coverage()

    def _build_coverage_setting(self, coverages_setting: Optional[Dict]) -> Dict:
        """Build coverage settings"""
        if coverages_setting is not None:
            return coverages_setting

        # Default settings
        base_cc_value = [5, 10, 19] if self.dataset == "cityscapes" else [50, 100, 150]

        if not self.fuzz:
            return {
                "NC": [0.25, 0.5, 0.75],
                "KMNC": [25, 50, 100, 1000],
                "SNAC": [None],
                "NBC": [None],
                "TKNC": [5, 10, 15],
                "TKNP": [10, 25, 50],
                "CC": base_cc_value,
                "NLC": [None],
            }
        else:
            return {
                "NC": 0.75,
                "KMNC": [100],
                "SNAC": [None],
                "NBC": [None],
                "TKNC": [15],
                "CC": [19] if self.dataset == "cityscapes" else [150],
                "TKNP": [25],
                "NLC": [None],
            }

    def _build_runner(self, config: str) -> tuple:
        """Build runner"""
        cfg = Config.fromfile(config)

        if self.mode == "train":
            cfg.val_dataloader.dataset.data_prefix = cfg.train_dataloader.dataset.data_prefix

        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(config))[0])
        cfg.load_from = self.checkpoint
        cfg.launcher = "none"

        runner = Runner.from_cfg(cfg)
        runner.model.cfg = cfg

        # Build processing pipeline
        pipeline = self._build_pipeline(cfg.test_pipeline)

        return runner, pipeline

    def _build_pipeline(self, test_pipeline: List) -> List:
        """Build processing pipeline"""
        pipeline = []
        excluded_types = {"LoadAnnotations", "GenerateEdge"}

        for pip in test_pipeline:
            if pip["type"] not in excluded_types and "cat_max_ratio" not in pip.keys():
                pipeline.append(pip)

        return pipeline

    def _get_output_size(self) -> tuple:
        """Get output sizes"""
        outputsizehook = OutputSizeHook(self)
        return outputsizehook.output_sizes, outputsizehook.layer_dict_keys

    def _build_coverage(self) -> Dict:
        """Build coverage metrics"""
        covers = {}
        for key, value in self.coverages_setting.items():
            cov = partial(
                getattr(coverage, key),
                model_name=self.model_name,
                layer_size_dict=self.output_sizes,
                device=self.device,
                save_path=self.coverage_save_path,
                TOOL_LOG_FILE_PATH=self.TOOL_LOG_FILE_PATH,
            )

            if isinstance(value, (int, type(None))):
                covers[f"{key}-{value}"] = [cov(threshold=value)]
            elif isinstance(value, list):
                for v in value:
                    covers[f"{key}-{v}"] = [cov(threshold=v)]

        return covers

    def recovery_coverages(self) -> None:
        """Recover coverage metrics"""
        self.coverages = self._build_coverage()
        for key, covers in tqdm(self.coverages.items(), desc="Recovery Coverage Metrics"):
            cover = covers[0]
            cover.recvoery_progress("test")

        self.print_coverages_info()

    def print_coverages_info(self, data:str=None) -> None:
        """Print coverage information"""
        re_data = {}
        for key, covers in self.coverages.items():
            cover = covers[0]
            logger.info(f"Current Coverage of {key} is {cover.current}.")
            re_data[key] = cover.current

        if data==None:
            data = ""

        self.print_times += 1
        torch.save(
            re_data,
            f"{self.data_save_path_prefix}/{data}-Use_for_print-{self.print_times}-{self.mode}-{Path(self.config).stem}-cov.pth",
        )
        return re_data

    def analyze_hooke_main(self) -> Dict:
        """Main analysis function"""
        self.coverages = self._build_coverage()

        save_layer_io_hook = SaveLayerInputOutputHook(self)
        self.runner.register_hook(save_layer_io_hook)
        self.print_coverages_info()
        metrics = self.runner.val()
        name = Path(self.config).stem

        return {name: metrics}

    def _process_build_coverage(self, exec_type: str) -> None:
        """Process build coverage"""
        self._rebuild_model(self.config)

        self.coverage_save_path = f"{self.data_save_path_prefix}/all_{exec_type}_cov"
        utility.build_path([self.coverage_save_path])

        res = self.analyze_hooke_main()
        config_name = Path(self.config).stem
        print(utility.dict_to_table(res[config_name]))

        logger.info(f"Results for {config_name}:")
        logger.info(f"  mIoU: {res[config_name]['mIoU']:.4f}")
        logger.info(f"  mAcc: {res[config_name]['mAcc']:.4f}")
        logger.info(f"  aAcc: {res[config_name]['aAcc']:.4f}")

        data = [
            config_name,
            res[config_name]["mIoU"],
            res[config_name]["mAcc"],
            res[config_name]["aAcc"],
        ]

        if len(self.coverages_setting) == 1:
            for key,value in self.coverages_setting.items():
                file_prefix = f"{key}-{value}"
        else:
            file_prefix="ALL"


        torch.save(
            data,
            f"{self.data_save_path_prefix}/{file_prefix}-{exec_type}-{config_name}-acc-iou.pth",
        )

        return {
            "mIoU": res[config_name]["mIoU"],
            "mAcc": res[config_name]["mAcc"],
            "aAcc": res[config_name]["aAcc"],
        }

    def _rebuild_model(self, config: str) -> None:
        """Rebuild model"""
        # Clean up old model
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "runner"):
            del self.runner

        # Build new model
        self.runner, _ = self._build_runner(config)
        self.model = self.runner.model
        self.layer_dict = utility.get_model_layers(self.model)

    def _process_file_for_select(self, save_path: str) -> Dict:
        """Process file for selection"""
        file_name = Path(self.config).stem
        self.coverage_save_path = f"{save_path}/{file_name}/"
        utility.build_path([self.coverage_save_path])

        # Parse coverage and hyperparameters
        parts = file_name.split("-")
        cov, hyper = parts[0], parts[1]
        hyper = None if hyper == "None" else float(hyper)

        self.coverages_setting = {cov: [hyper]}
        self._rebuild_model(self.config)

        return self.analyze_hooke_main()

    def pre_build_coverage_info(self, coverages_setting=None) -> None:
        """Pre-build coverage information"""
        self.mode = "train"
        if coverages_setting is not None:
            self.coverages_setting = coverages_setting
        self._process_build_coverage("train")

    def build_val_coverage_info(self) -> None:
        """Build validation coverage information"""
        self.mode = "test"
        re_iou = self._process_build_coverage("test")
        re_data = self.print_coverages_info()
        cal_cov.cal_ALL_cov(
            self.data_save_path_prefix,
            self.model_name,
            self.mode,
            self.coverages_setting,
        )
        re_iou.update(re_data)
        return re_iou

    def build_diverity_coverage_info(self) -> None:
        """Build diversity coverage information"""
        self.mode = "diversity"
        self._process_diversity_coverage("diversity")
        self.print_coverages_info()
        df = cal_cov.cal_ALL_cov(
            self.data_save_path_prefix,
            self.model_name,
            self.mode,
            self.coverages_setting,
        )
        config_name = Path(self.config).stem
        df.to_csv(
            f"{self.data_save_path_prefix}/ALL-{self.mode}-{config_name}_cov.csv",
            index=False,
        )

    def _process_diversity_coverage(self, exec_type: str) -> None:
        """Process diversity coverage"""
        self._rebuild_model(self.config)

        self.coverage_save_path = f"{self.data_save_path_prefix}/all_{exec_type}_cov"
        utility.build_path([self.coverage_save_path])

        res = self.analyze_hooke_main()
        config_name = Path(self.config).stem

        print(utility.dict_to_table(res[config_name]))

        logger.info(f"Diversity results for {config_name}:")
        logger.info(f"  mIoU: {res[config_name]['mIoU']:.4f}")
        logger.info(f"  mAcc: {res[config_name]['mAcc']:.4f}")
        logger.info(f"  aAcc: {res[config_name]['aAcc']:.4f}")

        torch.save(
            res,
            f"{self.data_save_path_prefix}/Diversity-ALL-{exec_type}-{config_name}-acc-iou.pth",
        )

    def build_coverage_info_for_fuzz(self, config: str) -> None:
        """Build coverage information for fuzzing"""
        self.mode = "fuzz_test_files"
        self.config = config
        self._process_build_coverage("fuzz_test_files")
        self.print_coverages_info()

    def select_files(self, func: Optional[callable] = None) -> None:
        """Select files"""
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
            func,
        )

    def build_select_coverage_info(self) -> None:
        """Build selection coverage information"""
        self.mode = "choice_test"
        self.seclect_file_path = (
            f"{self.dataset_path_prefix}/diversity/{self.model_name}/config0515"
        )
        files = utility.get_files(self.seclect_file_path)
        save_path = f"{self.data_save_path_prefix}/select_cov"
        if self.cov_select_type is not None:
            save_path = f"{save_path}/add_select"

        utility.build_path([save_path])

        self.ALL_FILE_NUM = len(files)

        results = []
        check_path = f"{self.data_save_path_prefix}/Select-acc-iou_Original.pth"
        logger.info(f"Check path: {check_path}")
        if os.path.exists(check_path):
            logger.info(f"File already exists, skipping coverage calculation")
        else:
            for idx in tqdm(
                range(len(files)),
                desc="Executing collection of coverage metrics for various datasets",
            ):
                self.config = files[idx]
                results.append(self._process_file_for_select(save_path))

        # Analyze results
        analyse.analysis_iou(
            results,
            self.data_save_path_prefix,
            self.model_name,
            self.coverages_setting,
            cov_select_type=self.cov_select_type,
        )
        self.coverages_setting = self._build_coverage_setting(None)
        cal_cov.cal_select_cov(
            self.data_save_path_prefix,
            self.coverages_setting,
            self.model_name,
            self.mode,
        )

    def cal_diversity_info(self) -> None:
        """Calculate diversity information"""
        self.coverages = self._build_coverage()
        cal_diversity.cal_diversity(
            self.dataset,
            self.dataset_path_prefix,
            self.model_name,
            self.coverages_setting,
            len(self.coverages),
            self.device,
            self.data_save_path_prefix,
        )

    def analyse_file(self) -> None:
        """Analyze file"""
        analyse.analyse_file(self.model_name, self.data_save_path_prefix)

    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "runner"):
            del self.runner
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def remove_files(self) -> None:
        """Remove files"""
        import shutil

        shutil.rmtree(self.fuzz_save_path_prefix)

    def get_add_cover_info(self) -> None:
        """Get additional coverage information"""
        self._rebuild_model(self.config)
        save_path = f"{self.data_save_path_prefix}/all_test_cov"

        flag = False
        path = f"{self.dataset_path_prefix}/diversity/{self.model_name}/config0515/"
        if os.path.exists(path):
            logger.info("Coverage information already exists, skipping")
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
                method="add",
            )


class TorchCoverageTest:
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_SEED = 42

    def __init__(
            self,
            model_name: str,
            dataset: str,
            model_type: str,
            model_path: str,
            dataset_path_prefix: str = "",
            transforms_config: Optional[Dict] = None,
            mode: str = "test",
            coverages_setting: Optional[Dict] = None,
            fuzz: bool = False,
            num_classes: int = 1000,
            input_size: tuple = (224, 224)
    ):
        """
        Initialize coverage test class for PyTorch models

        Args:
            model_name: Model name
            dataset: Dataset name
            model_type: Model type
            model_path: Model file path (.pth, .pt)
            dataset_path_prefix: Dataset path prefix
            transforms_config: Data transformation configuration
            mode: Run mode ("test", "train", "choice_test")
            coverages_setting: Coverage settings
            fuzz: Whether to use fuzzing
            num_classes: Number of classes
            input_size: Input size
        """
        # Model settings
        self.coveraeg_type = "torch"
        self.model_name = model_name
        self.dataset = dataset
        self.model_path = model_path
        self.device = self._get_device()
        self.mode = mode
        self.model_type = model_type
        self.fuzz = fuzz
        self.num_classes = num_classes
        self.input_size = input_size

        # Load model
        self.model = self._load_model()
        self.model.eval()
        self.transforms = None

        # Path settings
        self.dataset_path_prefix = dataset_path_prefix
        self._setup_paths()

        # Get output sizes and layer names
        self.layer_dict = utility.get_model_layers(
            self.model, self.TOOL_LOG_FILE_PATH, self.model_name
        )
        self.output_sizes, self.layer_dict_keys = self._get_output_size()

        # Coverage settings
        self.coverages_setting = self._build_coverage_setting(coverages_setting)
        logger.info(f"Coverage settings: {self.coverages_setting}")

        # Hook related (now managed by dedicated Hook class)
        self.save_layer_io_hook = None

        # Other settings
        self.seed = self.DEFAULT_SEED
        self.num_workers = self.DEFAULT_NUM_WORKERS
        self.batch_size = self.DEFAULT_BATCH_SIZE

    def _get_device(self) -> torch.device:
        """Get device"""
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _load_model(self) -> nn.Module:
        """Load model"""
        try:
            # If model class is provided, instantiate it and then load weights
            import torchvision.models as models
            model = getattr(models, self.model_name)(pretrained=False)
            model.load_state_dict(torch.load(self.model_path))
            model = model.to(self.device)
            logger.info(f"Model successfully loaded to {self.device}")
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _setup_paths(self) -> None:
        """Setup paths"""
        self.data_save_path_prefix = f"./output-data-0610/{self.dataset}/{self.model_type}/{self.model_name}"
        self.TOOL_LOG_FILE_PATH = f"./temp/{self.dataset}/{self.model_type}/{self.model_name}"
        utility.build_path([self.data_save_path_prefix, self.TOOL_LOG_FILE_PATH])

    def _build_coverage_setting(self, coverages_setting: Optional[Dict]) -> Dict:
        """Build coverage settings"""
        if coverages_setting is not None:
            return coverages_setting

        # Default settings
        base_cc_value = 19 if self.dataset == 'cityscapes' else 150

        if not self.fuzz:
            return {
                "NC": [0.25, 0.5, 0.75],
                "KMNC": [25, 50, 100, 1000],
                'SNAC': [None],
                'NBC': [None],
                'TKNC': [5, 10, 15],
                'TKNP': [10, 25, 50],
                'CC': [5, 10, base_cc_value],
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

    def _get_output_size(self) -> tuple:
        """Get output sizes"""
        outputsizehook = OutputSizeHook(self)
        return outputsizehook.output_sizes, outputsizehook.layer_dict_keys

    def _build_coverage(self) -> Dict:
        """Build coverage metrics"""
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
        """Recover coverage metrics"""
        for key, covers in tqdm(self.coverages.items(), desc="Recovery Coverage Metrics"):
            cover = covers[0]
            cover.recvoery_progress("test")

    def analyze_hooke_main(self, dataloader: DataLoader) -> Dict:
        """Main analysis function"""
        self.coverages = self._build_coverage()

        # Use new SaveLayerInputOutputHook
        save_layer_io_hook = SaveLayerInputOutputHook(self)

        # Process dataloader and collect coverage information
        save_layer_io_hook.before_run(self)
        save_layer_io_hook.before_val(dataloader)
        correct = 0
        total = 0
        for idx, data in enumerate(dataloader):
            with torch.no_grad():
                images, labels = data[0], data[1]
                images = images.to(self.device)
                labels = labels.to(self.device)
                save_layer_io_hook.before_val_iter(self.model, idx, images)
                outputs = self.model(images)
                save_layer_io_hook.after_val_iter(self.model, idx, images)
                # Get prediction results
                _, predicted = torch.max(outputs.data, 1)
                total += data[1].size(0)
                correct += (predicted == labels).sum().item()

        save_layer_io_hook.after_val(self.model)
        save_layer_io_hook.after_run(self.model)

        # Evaluate model performance
        accuracy = 100 * correct / total
        name = Path(self.model_path).stem
        logger.info(f'\nAccuracy of the model on the test images: {accuracy:.2f}%\n')

        return {name: accuracy}

    def _process_build_coverage(self, exec_type: str, dataloader: DataLoader) -> None:
        """Process build coverage"""
        self.coverage_save_path = f"{self.data_save_path_prefix}/all_{exec_type}_cov"
        utility.build_path([self.coverage_save_path])

        res = self.analyze_hooke_main(dataloader)
        config_name = Path(self.model_path).stem

    def pre_build_coverage_info(self, train_dataloader: DataLoader) -> None:
        """Pre-build coverage information"""
        self.mode = "train"
        self._process_build_coverage("train", train_dataloader)

    def build_val_coverage_info(self, val_dataloader: DataLoader) -> None:
        """Build validation coverage information"""
        self.mode = "test"
        self.coverages = self._build_coverage()
        self._process_build_coverage("test", val_dataloader)