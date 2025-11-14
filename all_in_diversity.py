import multiprocessing.spawn
import torch
import os
from rich.text import Text
from rich.rule import Rule
from rich.console import Console
import logging
import warnings
import multiprocessing
import utility1 as utility
from config1 import CoverageTest
from pathlib import Path

console = Console()
logging.disable(logging.CRITICAL)
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(profile="default")  # 避免张量输出过长



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    max_workers = 1
    
    DATASETS = [[],["ade20k"],["cityscapes"],["ade20k","cityscapes"]]
    MODEL_TYPE = ["","CNN","Transformer","Other"]

    datasets_idx = 3
    
    all_data = []
    all_data_div = []

    for model_type_idx in [1,2,3]:
        model_type = MODEL_TYPE[model_type_idx]
        console.print(Text(f"\nCurrent Model type is {model_type}.\n\n\n", style="red"))
        
        for dataset in DATASETS[datasets_idx]:
            model_infos,dataset_path_prefix = utility.get_file_device(dataset, model_type)
            for model_info in model_infos:
                model_name,config,checkpoint = model_info[0],model_info[1],model_info[2]
                base_cc_value = [5,10,19] if dataset == 'cityscapes' else [50,100,150]

                coverages_settings =  {
                    "NC": [0.25, 0.5, 0.75],
                    "KMNC": [25, 50, 100, 1000],
                    'SNAC': [None],
                    'NBC': [None],
                    'TKNC': [5, 10, 15],
                    'TKNP': [10, 25, 50],
                    'CC': base_cc_value,
                    'NLC': [None],
                }
                
                # if model_type == "CNN" and dataset == "ade20k":
                #     continue
                
                # if model_name in ["DeepLabV3Plus-R50-ade20k"]:
                #     continue
                # if model_name not in ["Segmenter-Vit_t-ade20k"]:


                for div_type in ["class", "class_new", "pixel", "entropy", "fid", "is", "ssim", "lpips", "tce", "tie"]:
                    for div_type2 in ["top", "random", "bottom"]:
                        if dataset == "cityscapes":
                            prefix_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes"
                        else:
                            prefix_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/ADEChallengeData2016"
                            
                        config_file = f"{prefix_path}/diversity_five_dim/config/{model_name}_{dataset}_{div_type}_{div_type2}.py"
                        
                        console.print(Text(f"\nWe're going to start testing {config_file} now!", style="bold bright_cyan"))
                        console.print(Rule(style="dim cyan"))

                        testTool = CoverageTest(model_name, dataset, model_type, config_file, checkpoint, dataset_path_prefix, coverages_setting=coverages_settings,save_data_path="0919-iou-cov")
                        
                        path = f"{testTool.data_save_path_prefix}/ALL-test-{Path(testTool.config).stem}-cov.pth"
                        if os.path.exists(path):
                            continue
                        
                        print("testTool.data_save_path_prefix = ", testTool.data_save_path_prefix)
                        utility.seed_everything(testTool.seed)

                        # Step 2: 对全部val数据进行覆盖检测
                        console.print(Text("Step 2. Perform coverage detection on all val data.", style="bold green"))
                        re_data = testTool.build_val_coverage_info()
                        
                        # data_save_path_prefix = testTool.data_save_path_prefix
                        # device = testTool.device

                        # torch.save(re_data, f"{data_save_path_prefix}/{div_type}-{div_type2}-iou-cover.pth")
                        # all_data.append((model_name, dataset, model_type, div_type, div_type2, re_data))


                # else:
                #     testTool = CoverageTest(model_name, dataset, model_type, config, checkpoint, dataset_path_prefix, coverages_setting=coverages_settings)
                #     data_save_path_prefix = testTool.data_save_path_prefix
                #     device = testTool.device
                            
                # Step 5: 对3个进行多样性的计算
                # import cal_diversity
                # console.print(Text(f"Step 4. Calculate diversity", style="bold green"))
                # re_data = cal_diversity.cal_diversity_for_div(dataset, dataset_path_prefix, model_name,  device, data_save_path_prefix)
                # torch.save(re_data, f"{data_save_path_prefix}/{div_type}-{div_type2}-div.pth")
                # all_data_div.append((model_name, dataset, model_type, div_type, div_type2, re_data))
                
    
    # torch.save(all_data, f"./all_in_diversity-iou-acc.pth")        
    # torch.save(all_data_div, f"./all_in_diversity_div.pth")   