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
    
    for model_type_idx in [3]:
        model_type = MODEL_TYPE[model_type_idx]
        console.print(Text(f"\nCurrent Model type is {model_type}.\n\n\n", style="red"))
        
        for dataset in DATASETS[datasets_idx]:
            model_infos,dataset_path_prefix = utility.get_file_device(dataset, model_type)
            for model_info in model_infos:
                model_name,config,checkpoint = model_info[0],model_info[1],model_info[2]
                base_cc_value = 19 if dataset == 'cityscapes' else 150

                coverages_settings =  {
                    "NC": [0.25, 0.5, 0.75],
                    "KMNC": [25, 50, 100, 1000],
                    'SNAC': [None],
                    'NBC': [None],
                    'TKNC': [5, 10, 15],
                    'TKNP': [10, 25, 50],
                    'CC': [5, 10, base_cc_value],
                    'NLC': [None],
                }
                
                # for key,value in coverages_settings.items():
                    # if model_type in ["Transformer","Other"]:
                    #     continue
                      
                # print(f"============================== {key}-{value} ==============================================")
                # coverages_setting = {key:value}
                console.print(Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
                console.print(Rule(style="dim cyan"))

                testTool = CoverageTest(model_name, dataset, model_type, config, checkpoint, dataset_path_prefix, cov_select_type="add",coverages_setting=coverages_settings)
                    
                utility.seed_everything(testTool.seed)

                try:
                    testTool.recovery_coverages()
                except Exception as e:
                # #     print(f"Error during recovery: {e}")
                # #     path = f"./temp/{dataset}/{model_type}/{model_name}/coverages"
                    if model_name in ["Upernet_Convnext_base-ade20k","SegFormer-Mit_b0-cityscapes","Segnext_Mscan-b_1-cityscapes","Segmenter-Vit_t-cityscapes",
                                        "Segmenter-Vit_t-cityscapes","Upernet-Deit_s16-cityscapes"]:
                        coverages_settings =  {
                                            "KMNC": [25, 50, 100, 1000],
                                            'SNAC': [None],
                                            'NBC': [None],
                                            'TKNC': [5, 10, 15],
                                            'TKNP': [10, 25, 50],
                                            'CC': [5, 10, base_cc_value],
                                            'NLC': [None],
                                        }
        
                        console.print(Text("Step 1. Preliminary construction of coverage measurement information.", style="bold green"))
                        testTool.pre_build_coverage_info(coverages_settings)
                        
                    # Step 2: 对全部val数据进行覆盖检测
                    console.print(Text("Step 2. Perform coverage detection on all val data.", style="bold green"))
                    testTool.build_val_coverage_info()

                console.print(Text("Step 2. Add Metric.", style="bold green"))
                testTool.get_add_cover_info()
               

                # # Step 4: 对3个进行覆盖度量的计算
                # console.print(Text(f"Step 3. Calculate coverage metrics for three {testTool.choice_samples}s.", style="bold green"))
                # testTool.build_select_coverage_info()

                # # Step 5: 对3个进行多样性的计算
                # console.print(Text(f"Step 5. Calculate diversity for three {testTool.choice_samples}s", style="bold green"))
                # testTool.cal_diversity_info()
                
                # # Step 6: 文件分析
                # console.print(Text(f"Step 6. Analyze the file and summarize it into 3 forms", style="bold green"))
                # testTool.analyse_file()