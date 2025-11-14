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
    max_workers = 1
    
    DATASETS = [[],["ade20k"],["cityscapes"],["ade20k","cityscapes"]]
    MODEL_TYPE = ["","CNN","Transformer","Other"]

    # model_type_idx = int(input("Please enter the type of model you want to perform (1. CNN, 2. Transformer, 3. Other):"))
    
    
    # datasets_idx = int(input("Please enter the serial number of the dataset to be selected (1. [ade20k], 2. [cityscapes], 3. [ade20k,cityscapes]):"))
    datasets_idx = 3
    
    for model_type_idx in [1,2,3]:
        model_type = MODEL_TYPE[model_type_idx]
        
        console.print(Text(f"\nCurrent Model type is {model_type}.\n\n\n", style="black"))
        
        for dataset in DATASETS[datasets_idx]:
            if model_type != "Other" and dataset != "ade20k":
                continue
            
            model_infos,dataset_path_prefix = utility.get_file_device(dataset, model_type)
            for model_info in model_infos:
                model_name,config,checkpoint = model_info[0],model_info[1],model_info[2]
                
                if model_name != "Segnext_Mscan-b_1-cityscapes":
                    continue
                
                print("============================================================")
                print( "Current testing model is:", model_name)
                
                print(model_type,dataset,model_name,dataset_path_prefix)
                
                testTool = CoverageTest(model_name, dataset, model_type, config, checkpoint, dataset_path_prefix,save_data_path="0908-all")

                utility.seed_everything(testTool.seed)

                console.print(Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
                console.print(Rule(style="dim cyan"))

                # Step 1: 初步构建度量信息
                # console.print(Text("Step 1. Preliminary construction of coverage measurement information.", style="bold green"))
                # testTool.coverages_setting = {
                #     "KMNC": [25, 50, 100, 1000],
                #     'SNAC': [None],
                #     'NBC': [None],
                # }
                # testTool.pre_build_coverage_info()
            
                # Step 2: 对全部val数据进行覆盖检测
                console.print(Text("Step 2. Perform coverage detection on all val data.", style="bold green"))
                testTool.build_val_coverage_info()

                # Step 3: 选出3个 or 移动对应的文件
                console.print(Text(f"Step 3. Choice the {testTool.choice_samples} files of val dataset (bottom/top/random).", style="bold green"))
                testTool.select_files()

                # Step 4: 对3个进行覆盖度量的计算
                console.print(Text(f"Step 4. Calculate coverage metrics for three {testTool.choice_samples}s.", style="bold green"))
                testTool.build_select_coverage_info()

                # Step 5: 对3个进行多样性的计算
                # if model_name in ["DeepLabV3Plus-R50-ade20k"]:
                console.print(Text(f"Step 5. Calculate diversity for three {testTool.choice_samples}s", style="bold green"))
                testTool.cal_diversity_info()
                # else:
                    # import cal_diversity
                    # testTool.coverages = testTool._build_coverage()
                    # print("     -> cal_diversity.only_for_ssim")
                    # cal_diversity.only_for_ssim(testTool.dataset, testTool.dataset_path_prefix, testTool.model_name, testTool.coverages_setting, len(testTool.coverages), testTool.device, testTool.data_save_path_prefix )
                
                # Step 6: 文件分析
                # # 分析结果
                import analyse
                # print("     -> analyse.analysis_iou")
                analyse.analysis_iou([], testTool.data_save_path_prefix, testTool.model_name, testTool.coverages_setting,cov_select_type = testTool.cov_select_type)
                console.print(Text(f"Step 6. Analyze the file and summarize it into 3 forms", style="bold green"))
                testTool.analyse_file()