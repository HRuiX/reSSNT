# import utility
# from fuzzer_core import Fuzzer
# import torch
# from img_conv import build_img_np_list
# import torch
# from rich.text import Text
# from rich.rule import Rule
# from rich.console import Console
# import logging
# import warnings
# import multiprocessing
# import utility
# from config import CoverageTest
#
# console = Console()
# logging.disable(logging.CRITICAL)
# multiprocessing.set_start_method('spawn', force=True)
# warnings.filterwarnings("ignore", category=UserWarning)
# torch.set_printoptions(profile="default")  # 避免张量输出过长
#
# if __name__ == '__main__':
#
#     DATASETS = [[], ["ade20k"], ["cityscapes"], ["ade20k", "cityscapes"]]
#     MODEL_TYPE = ["", "CNN", "Transformer", "Other"]
#
#     datasets_idx = 3
#     for model_type_idx in [1, 2, 3]:
#         model_type = MODEL_TYPE[model_type_idx]
#         for dataset in DATASETS[datasets_idx]:
#             model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
#             for model_info in model_infos:
#                 model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
#                 coverages_settings =  {
#                         "NC": [0.75],
#                         "KMNC": [100],
#                         'SNAC': [None],
#                         'NBC': [None],
#                         'TKNC': [15],
#                         'CC': [19] if dataset == 'cityscapes' else [150],
#                         'TKNP': [25],
#                         'NLC': [None],
#                     }
#
#                 if model_type == "CNN" or (model_type == "Transformer" and dataset =="ade20k"):
#                     continue
#
#                 for key,value in coverages_settings.items():
#                     for v in value:
#
#                         if model_name == "Mask2Former-Swin_S-cityscapes":
#                             if key in ["NC","KMNC","SNAC","NBC","TKNC","CC"]:
#                                 continue
#
#                         coverages_setting = {key:[v]}
#                         console.print(Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
#                         console.print(Rule(style="dim cyan"))
#
#                         testTool = CoverageTest(model_name, dataset, model_type, config, checkpoint, dataset_path_prefix,
#                                                 fuzz=True, coverages_setting=coverages_setting)
#
#                         # Step 2: 对全部val数据进行覆盖检测
#                         console.print(Text("Step 2. Perform coverage detection on all val data.", style="bold green"))
#                         testTool.build_val_coverage_info()
#
#                         data_list = build_img_np_list(config)
#                         print(f"现在我们准备变异的种子集合共有 {len(data_list)} 个种子")
#
#                         engine = Fuzzer(testTool)
#                         engine.run(data_list)
#                         engine.exit()


import utility
from fuzz import Fuzzer
import torch
from img_conv import build_img_np_list
import torch
from rich.text import Text
from rich.rule import Rule
from rich.console import Console
import logging
import warnings
import multiprocessing
import utility
from config import CoverageTest

console = Console()
logging.disable(logging.CRITICAL)
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(profile="default")  # Avoid excessively long tensor output

if __name__ == '__main__':

    dataset = "cityscapes"
    model_type = "Transformer"
    model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
    num_classes = 150 if dataset == "ade20k" else 19
    coverages_settings = {
        # "NC": 0.75,
        # "KMNC": 100,
        # 'SNAC': None,
        # 'NBC': None,
        # 'TKNC': 15,
        # 'CC': 19 if dataset == 'cityscapes' else 150,
        # 'TKNP': 25,
        'NLC': None,
    }

    for model_info in model_infos:
        model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
        for key, value in coverages_settings.items():
            console.print(
                Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
            console.print(Rule(style="dim cyan"))

            data_list = build_img_np_list(config)
            console.print(f"Prepared {len(data_list)} seed images for mutation")

            engine = Fuzzer(model_name, config, checkpoint, num_classes, [key,value], dataset=dataset,model_type=model_type, save_data_path="1104-use-new")
            engine.run(data_list)
            engine.exit()