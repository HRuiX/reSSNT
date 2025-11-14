import torch
from rich.text import Text
from rich.rule import Rule
from rich.console import Console
import logging
import warnings
import multiprocessing
import utility1 as utility
from config1 import CoverageTest

console = Console()
logging.disable(logging.CRITICAL)
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(profile="default")  # 避免张量输出过长

if __name__ == "__main__":
    max_workers = 1

    # DATASETS = [[], ["ade20k"], ["cityscapes"], ["ade20k", "cityscapes"]]
    # MODEL_TYPE = ["", "CNN", "Transformer", "Other"]
    #
    # # model_type_idx = int(input("Please enter the type of model you want to perform (1. CNN, 2. Transformer, 3. Other):"))
    #
    # # datasets_idx = int(input("Please enter the serial number of the dataset to be selected (1. [ade20k], 2. [cityscapes], 3. [ade20k,cityscapes]):"))
    # datasets_idx = 3
    #
    # for model_type_idx in [1, 2, 3]:
    #     model_type = MODEL_TYPE[model_type_idx]
    #
    #     console.print(Text(f"\nCurrent Model type is {model_type}.\n\n\n", style="black"))
    #
    #     for dataset in DATASETS[datasets_idx]:
    #         model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
    dataset = "cityscapes"
    model_type = "Transformer"
    model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
    num_classes = 150 if dataset == "ade20k" else 19

    for model_info in model_infos:
        model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
        for model_info in model_infos:
            model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]

            print("============================================================")
            print("Current testing model is:", model_name)

            print(model_type, dataset, model_name, dataset_path_prefix)

            testTool = CoverageTest(model_name, dataset, model_type, config, checkpoint, dataset_path_prefix)
            utility.seed_everything(testTool.seed)

            console.print(Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
            console.print(Rule(style="dim cyan"))

            # Step 1: 初步构建度量信息
            console.print(Text("Step 1. Preliminary construction of coverage measurement information.", style="bold green"))
            testTool.coverages_setting = {
                "KMNC": [100],
                'SNAC': [None],
                'NBC': [None],
                "ADC":[100]
            }
            testTool.pre_build_coverage_info()
