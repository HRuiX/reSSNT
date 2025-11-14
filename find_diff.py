import torch
import os
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
torch.set_printoptions(profile="default")  # 避免张量输出过长


if __name__ == "__main__":
    max_workers = 1
    
    DATASETS = [[],["ade20k"],["cityscapes"],["ade20k","cityscapes"]]
    MODEL_TYPE = ["","CNN","Transformer","Other"]
    
    
    model_name = f"DeepLabV3Plus-R50-cityscapes"
    config = "./ckpt/CNN/cityscapes/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py"
    checkpoint = "./ckpt/CNN/cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth"
    dataset_path_prefix = "./data/cityscapes/"
    model_type = "CNN"
    dataset = "cityscapes"
    coverages_setting={"NC":[0.5]}
    
    testTool = CoverageTest(model_name, dataset, model_type, config, checkpoint, dataset_path_prefix,
                                                fuzz=True, coverages_setting=coverages_setting)
    
    utility.seed_everything(testTool.seed)

    console.print(Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
    console.print(Rule(style="dim cyan"))

    try:
        testTool.recovery_coverages()
    except Exception as e:
        print(f"Error during recovery: {e}")
        # Step 1: 初步构建度量信息
        console.print(Text("Step 1. Preliminary construction of coverage measurement information.", style="bold green"))
        testTool.pre_build_coverage_info()
        
        # Step 2: 对全部val数据进行覆盖检测
        console.print(Text("Step 2. Perform coverage detection on all val data.", style="bold green"))
        testTool.build_val_coverage_info()
    
                
    for seed in [0,4,12,32,42,None]:
        console.print(Text(f"Step X. Perform coverage detection on all Fuzz data for seed {seed}.", style="bold green"))
        config = f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-fuzz-data-0615/NC-05/test-{seed}/test-{seed}.py"
        testTool.build_coverage_info_for_fuzz(config)
    
    
    
    
    
    
    
    
    
   
    