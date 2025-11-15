# # import torch
# # import os
# # from rich.text import Text
# # from rich.rule import Rule
# # from rich.console import Console
# # import logging
# # import warnings
# # import multiprocessing
# # import utility
# # from config import CoverageTest
#
# # console = Console()
# # logging.disable(logging.CRITICAL)
# # multiprocessing.set_start_method('spawn', force=True)
# # warnings.filterwarnings("ignore", category=UserWarning)
# # torch.set_printoptions(profile="default")  # 避免张量输出过长
# # # os.environ["NO_ALBUMENTATIONS_UPDATE"] = 1
#
# # dataset = "ade20k"
# # model_name = f"DeepLabV3Plus-R50-{dataset}"
# # model_type = "CNN"
# # config = "./ckpt/CNN/ade20k/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py"
# # checkpoint = "./ckpt/CNN/ade20k/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth"
# # dataset_path_prefix = "./data/ADEChallengeData2016/"
#
# # testTool = CoverageTest(model_name, dataset, model_type, config, checkpoint, dataset_path_prefix)
#
# # utility.seed_everything(testTool.seed)
#
# # console.print(Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
# # console.print(Rule(style="dim cyan"))
#
# # # Step 1: 初步构建度量信息
# # console.print(Text("Step 1. Preliminary construction of coverage measurement information.", style="bold green"))
# # testTool.pre_build_coverage_info()
#
#
# # # Step 2: 对全部val数据进行覆盖检测
# # console.print(Text("Step 2. Perform coverage detection on all val data.", style="bold green"))
# # testTool.build_val_coverage_info()
#
#
# # # Step 3: 选出3个 or 移动对应的文件
# # console.print(Text(f"Step 3. Choice the {testTool.choice_samples} files of val dataset (bottom/top/random).", style="bold green"))
# # testTool.select_files()
#
# # # Step 4: 对3个进行覆盖度量的计算
# # console.print(Text(f"Step 4. Calculate coverage metrics for three {testTool.choice_samples}s.", style="bold green"))
# # testTool.build_select_coverage_info()
#
# # # Step 5: 对3个进行多样性的计算
# # console.print(Text(f"Step 5. Calculate diversity for three {testTool.choice_samples}s", style="bold green"))
# # testTool.cal_diversity_info()
#
# # # Step 6: 文件分析
# # console.print(Text(f"Step 6. Analyze the file and summarize it into 3 forms", style="bold green"))
# # testTool.analyse_file()
#
#
# import os
# import utility
# import shutil
# import pandas as pd
# def count_folders(path):
#     count = 0
#     for item in os.listdir(path):
#         if os.path.isdir(os.path.join(path, item)):
#             count += 1
#     return count
#
# def method1_os_walk(folder_path):
#     all_files = []
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             all_files.append(file_path)
#
#     return all_files
#
#
# def count_files_simple(directory_path):
#     """
#     简单计算指定目录中的文件数量（不包括子目录）
#     """
#     try:
#         files = [f for f in os.listdir(directory_path)
#                 if os.path.isfile(os.path.join(directory_path, f))]
#         return len(files)
#     except OSError as e:
#         print(f"错误：无法访问目录 {directory_path}: {e}")
#         return 0
#
# from config import CoverageTest
# DATASETS = [[], ["ade20k"], ["cityscapes"], ["ade20k", "cityscapes"]]
# MODEL_TYPE = ["", "CNN", "Transformer", "Other"]
# datasets_idx = 3
# for model_type_idx in [1,2,3]:
#         model_type = MODEL_TYPE[model_type_idx]
#         for dataset in DATASETS[datasets_idx]:
#             model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
#             for model_info in model_infos:
#                 model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
#                 coverages_settings =  {
#                         "NC": [0.5, 0.75],
#                         "KMNC": [100],
#                         'SNAC': [None],
#                         'NBC': [None],
#                         'TKNC': [15],
#                         'CC': [19] if dataset == 'cityscapes' else [150],
#                         'TKNP': [25],
#                         'NLC': [None],
#                     }
#                 for ke,value in coverages_settings.items():
#                     for v in value:
#                         if "." in str(v):
#                             v = str(v).replace(".", "")
#
#                         key = f"{ke}-{v}"
#                         file_record = []
#                         prefix_path = f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-fuzz-data-0714/{key}/{dataset}/{model_type}/fuzz_test/"
#                         print(f"处理路径: {prefix_path}")
#                         if not os.path.exists(prefix_path):
#                                 continue
#
#                         folder_count = count_folders(prefix_path)
#                         print(f"文件夹数量: {folder_count}")
#
#                         epoch_cnt = 0
#                         for epoch in range(folder_count):
#                             epoch_key = f"epoch-{epoch}"
#                             if dataset == "ade20k":
#                                 path = f"{prefix_path}/epoch-{epoch}/muta/ADEChallengeData2016/images/validation"
#                             else:
#                                 path = f"{prefix_path}/epoch-{epoch}/muta/cityscapes/leftImg8bit/val"
#
#                             if not os.path.exists(path):
#                                 # print(f"路径不存在: {path}")
#                                 continue
#
#                             files = method1_os_walk(path)
#                             epoch_cnt += len(files)
#                             # for file in files:
#                             #     name = file.split("/")[-1]
#                             #     file_record.append([key, epoch_key, "ori", name])
#                             shutil.rmtree(f"{prefix_path}/epoch-{epoch}")
#                             print(f"成功删除文件夹: {prefix_path}/epoch-{epoch}/muta")
#                         print(f"Model Type {model_type}, model Name {model_name}, Covergae {key} has files {epoch_cnt}")
#                         if file_record == []:
#                             print(f"没有找到任何文件在路径: {prefix_path}")
#                             continue
#                         else:
#                             print(f"找到 {len(file_record)} 个文件在路径: {prefix_path}")
#                             file_record = pd.DataFrame(file_record, columns=["Coverage", "Epoch", "Type", "Name"])
#                             file_record.to_csv(f"{prefix_path}/file_muta_record.csv", index=False, mode='a', header=False)
#

import torch
import mmcv
import numpy as np
import os
from mmseg.apis import inference_model, init_model

# 配置文件 (来自 mim download)
config_file = '/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py'
# 权重文件 (来自 mim download)
checkpoint_file = '/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024_20221127_143802-9ab177f6.pth'
# 检查设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("正在加载模型...")
# 从配置和权重文件中初始化模型
model = init_model(config_file, checkpoint_file, device=device)
print("模型加载完毕。")

# 你的测试图像
image_path = '/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png'
# 你的真实分割图 (Ground Truth)
gt_path = '/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png'

print("正在加载图像和 Ground Truth...")
# 1. 加载你的输入图像
img = mmcv.imread(image_path)

inference_model(model, img)
