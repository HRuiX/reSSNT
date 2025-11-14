# import json
# import os
# import glob
# import random
# random.seed(42)
# import shutil
# from pathlib import Path

# def get_diveristy_file_list(div_type,dataset):
#     path = f"/home/ictt/xhr/code/DNNTesting/reSSNT/diversity_results_dict_format/{dataset}/{div_type}_path_to_score.json"
#     dat =  json.load(open(path))
#     topdat = sorted(dat.items(), key=lambda x: x[1], reverse=True)
#     tda = [os.path.basename(i[0])for i in topdat[:100]]
#     topdats = [i[0] for i in topdat[:100]]

#     botdat = sorted(dat.items(), key=lambda x: x[1], reverse=False)
#     bda = [os.path.basename(i[0])for i in botdat[:100]]
#     botdats = [i[0] for i in botdat[:100]]

#     randats = list(dat.keys())
#     randats = random.sample(randats,100)

#     return {"top":topdats,"bottom":botdats,"random":randats}

# def move_files_to_folder(file_paths, destination_folder):
#     dest_path = Path(destination_folder)
#     dest_path.mkdir(parents=True, exist_ok=True)
    
#     success_count = 0
#     failed_files = []
    
#     for file_path in file_paths:
#         source = Path(file_path)
#         try:
#             destination = dest_path / source.name
#             shutil.copy(str(source), str(destination))
#             success_count += 1
#         except Exception as e:
#             print(f"移动失败 {source.name}: {e}")
#             failed_files.append(file_path)
    
#     print(f"\n总计: 成功移动 {success_count} 个文件")
#     return failed_files

# for div_type in ["fid","is","lpips","tce","tie"]:
#     for dataset in ["ade20k","cityscapes"]:
#         all_div_path = get_diveristy_file_list(div_type,dataset)
#         for key,value in all_div_path.items():
#             if dataset == "ade20k":
#                 target_path = f"/home/ictt/xhr/code/DNNTesting/reSSNT/data/ADEChallengeData2016/diversity_five_dim/diversity/{div_type}/{key}/ADEChallengeData2016/leftImg8bit/validation"
#             else:
#                 target_path = f"/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes/diversity_five_dim/diversity/{div_type}/{key}/cityscapes/leftImg8bit/val"
            
#             move_files_to_folder(value, target_path)



import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from mmengine.registry import init_default_scope
from mmseg.datasets import CityscapesDataset, ADE20KDataset
import random
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial 
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from image_diversity import ClipMetrics
from image_diversity import InceptionMetrics
import cv2
from pytorch_fid import fid_score
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import warnings
import logging
import imghdr
from pathlib import Path
from functools import lru_cache
import gc
from typing import List, Tuple, Dict, Any
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
import cal_diversity

console = Console()
logging.disable(logging.CRITICAL)
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(profile="default")  # 避免张量输出过长


try:
    from pytorch_msssim import ssim as torch_ssim
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    print("Warning: pytorch-msssim not found. Install with: pip install pytorch-msssim")
    HAS_PYTORCH_MSSSIM = False

def rgb_to_grayscale_gpu(image_batch):
    """GPU上将RGB图像转换为灰度图像"""
    # 使用标准RGB到灰度的转换权重
    weights = torch.tensor([0.299, 0.587, 0.114], device=image_batch.device).view(1, 3, 1, 1)
    return torch.sum(image_batch * weights, dim=1, keepdim=True)

def calculate_ssim_gpu_batch(img1_batch, img2_batch):
    """
    使用GPU批量计算SSIM
    img1_batch, img2_batch: (batch_size, channels, height, width)
    """
    if HAS_PYTORCH_MSSSIM:
        # 使用pytorch-msssim，支持GPU加速
        ssim_values = torch_ssim(img1_batch, img2_batch, data_range=1.0, size_average=False)
        return ssim_values
    else:
        # 回退到手动实现的GPU SSIM
        return manual_ssim_gpu(img1_batch, img2_batch)

def manual_ssim_gpu(img1_batch, img2_batch, window_size=11, sigma=1.5):
    """
    手动实现的GPU SSIM计算（如果没有pytorch-msssim）
    """
    device = img1_batch.device
    
    # 转换为灰度图像（如果是RGB）
    if img1_batch.shape[1] == 3:
        img1_batch = rgb_to_grayscale_gpu(img1_batch)
        img2_batch = rgb_to_grayscale_gpu(img2_batch)
    
    # 创建高斯窗口
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
    
    # SSIM常数
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    
    # 计算均值
    mu1 = F.conv2d(img1_batch, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2_batch, window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1_batch ** 2, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2_batch ** 2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1_batch * img2_batch, window, padding=window_size//2, groups=1) - mu1_mu2
    
    # SSIM计算
    numerator1 = 2 * mu1_mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2
    
    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    
    # 返回每个图像的平均SSIM值
    return ssim_map.mean(dim=[1, 2, 3])

def calculate_ssim_diversity_scores_cuda(dataset1: List, idx: int, batch_size: int = 128, device: str = 'cuda'):
    """
    进一步优化的CUDA SSIM计算，使用更高效的批处理
    """
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    device = torch.device(device)
    
    print(f"Loading {len(dataset1)} images to {device}...")
    
    # 预处理并转移到GPU
    gpu_images = []
    for i in range(len(dataset1)):
        img = dataset1[i]
        if not isinstance(img, torch.Tensor):
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if img.max() > 1.0:
            img = img / 255.0
            
        gpu_images.append(img.to(device))
    
    # 堆叠所有图像
    all_images = torch.stack(gpu_images)  # (N, C, H, W)
    del gpu_images  # 释放内存
    
    ssim_values = []
    n_images = all_images.shape[0]
    
    progress = tqdm(total=n_images * (n_images - 1), desc=f"Calculating SSIM on {device}")
    
    # 分块处理以节省内存
    chunk_size = min(batch_size, n_images)
    
    for i in range(0, n_images, chunk_size):
        chunk1 = all_images[i:i+chunk_size]  # (chunk_size, C, H, W)
        
        for j in range(0, n_images, chunk_size):
            chunk2 = all_images[j:j+chunk_size]  # (chunk_size, C, H, W)
            
            # 扩展维度进行批量比较
            chunk1_expanded = chunk1.unsqueeze(1)  # (chunk_size, 1, C, H, W)
            chunk2_expanded = chunk2.unsqueeze(0)  # (1, chunk_size, C, H, W)
            
            # 计算所有组合的SSIM
            for idx1 in range(chunk1.shape[0]):
                for idx2 in range(chunk2.shape[0]):
                    actual_i = i + idx1
                    actual_j = j + idx2
                    
                    if actual_i == actual_j:
                        progress.update(1)
                        continue
                    
                    img1 = chunk1[idx1:idx1+1]  # (1, C, H, W)
                    img2 = chunk2[idx2:idx2+1]  # (1, C, H, W)
                    
                    try:
                        ssim_val = calculate_ssim_gpu_batch(img1, img2)
                        ssim_values.append(ssim_val.item())
                    except Exception as e:
                        print(f"SSIM calculation error: {e}")
                        
                    progress.update(1)
    
    progress.close()
    
    # 清理GPU内存
    del all_images
    torch.cuda.empty_cache()
    
    mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
    return 1.0 - mean_ssim

# 替换原函数
def calculate_ssim_diversity_scores(dataset1: List, idx: int, batch_size: int = 64):
    """
    兼容性包装函数，自动选择CUDA或CPU版本
    """
    try:
        if torch.cuda.is_available():
            # 根据GPU内存选择批次大小
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 8:
                batch_size = min(batch_size, 64)
            else:
                batch_size = min(batch_size, 32)
                
            return calculate_ssim_diversity_scores_cuda(
                dataset1, idx, batch_size, device='cuda'
            )
        else:
            print("CUDA not available, using CPU version")
            return calculate_ssim_diversity_scores_cpu_original(dataset1, idx, batch_size)
    except Exception as e:
        print(f"CUDA version failed: {e}, falling back to CPU")
        return calculate_ssim_diversity_scores_cpu_original(dataset1, idx, batch_size)

def calculate_ssim_diversity_scores_cpu_original(dataset1: List, idx: int, batch_size: int = 128):
    """原始CPU版本作为备份"""
    from skimage.metrics import structural_similarity as ssim
    
    ssim_values = []
    
    progress = tqdm(total=len(dataset1) * (len(dataset1)-1), 
                   desc=f"   → Calculating {idx} SSIM Distance (CPU)", position=idx)
    
    for i in range(0, len(dataset1), batch_size):
        batch1 = dataset1[i:i + batch_size]
        
        for j in range(0, len(dataset1), batch_size):
            batch2 = dataset1[j:j + batch_size]
            
            for img1 in batch1:
                img1_np = img1.permute(1, 2, 0).numpy()
                
                for img2 in batch2:
                    if torch.equal(img1, img2):
                        continue
                    try:
                        img2_np = img2.permute(1, 2, 0).numpy()
                        ssim_val = ssim(img1_np, img2_np, data_range=1.0, 
                                    channel_axis=2, multichannel=True)
                        ssim_values.append(ssim_val)
                        
                    except Exception as e:
                        print(f"SSIM calculation error: {e}")
                        continue
                    finally:
                        progress.update()
    
    progress.close()
    mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
    return 1.0 - mean_ssim

# 其余代码保持不变...

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

VALID_IMG_FORMATS = {"jpeg", "png", "gif", "bmp", "tiff"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif"}

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning)
init_default_scope('mmseg')

def get_img_names(path: str) -> List[str]:
    """优化的图像文件获取函数，使用pathlib和扩展名过滤"""
    img_names = []
    path_obj = Path(path)
    
    for ext in VALID_EXTENSIONS:
        img_names.extend([str(p) for p in path_obj.rglob(f"*{ext}")])
        img_names.extend([str(p) for p in path_obj.rglob(f"*{ext.upper()}")])
    
    return img_names

class CustomDataset(Dataset):
    """优化的数据集类，支持GPU预加载"""
    def __init__(self, image_folder: str, transform=None, preload: bool = False, device: str = 'cpu'):
        self.image_folder = image_folder
        self.image_paths = get_img_names(image_folder)
        self.device = torch.device(device)
        self.transform = transform or transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor()
        ])
        
        # 根据可用内存决定是否预加载
        self.preload = preload and len(self.image_paths) < 1000
        
        if self.preload:
            with ThreadPoolExecutor(max_workers=4) as executor:
                images = list(executor.map(self._load_image, self.image_paths))
                self.images = [img.to(self.device) for img in images]
        else:
            self.images = None

    def _load_image(self, img_path: str) -> torch.Tensor:
        """加载单个图像文件并转换为tensor"""
        if isinstance(img_path, (list, tuple)):
            raise ValueError(f"Expected string path, got {type(img_path)}: {img_path}")
        
        image = Image.open(str(img_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.image_paths)))
            images = []
            for i in indices:
                if self.preload:
                    image = self.images[i]
                else:
                    image = self._load_image(self.image_paths[i])
                    if not self.preload and self.device.type == 'cuda':
                        image = image.to(self.device)
                images.append(image)
            return images
        else:
            if self.preload:
                image = self.images[idx]
            else:
                image = self._load_image(self.image_paths[idx])
                if not self.preload and self.device.type == 'cuda':
                    image = image.to(self.device)
            return image
        

# CUDA加速的主脚本
import torch
import gc
import pandas as pd
from pathlib import Path


    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 优化CUDA设置
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提升性能
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    max_workers = 1

    DATASETS = [[], ["ade20k"], ["cityscapes"], ["ade20k", "cityscapes"]]
    MODEL_TYPE = ["", "CNN", "Transformer", "Other"]

    datasets_idx = 3

    all_data_save = []
    all_data_div = []


    batch_size = 128

    for model_type_idx in [1, 2, 3]:
        model_type = MODEL_TYPE[model_type_idx]

        for dataset in DATASETS[datasets_idx]:
            
            model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
            
            for model_info in model_infos:
                model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
                all_data = []

                data_save_path_prefix = '/home/ictt/xhr/code/DNNTesting/reSSNT/output-data-None/files'
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                
                console.print(Text(f"Step 4. Calculate diversity", style="bold green"))
                # re_data = cal_diversity.cal_diversity_for_div(dataset, dataset_path_prefix, model_name,  device, data_save_path_prefix)
                # re_data["dataset"] = dataset
                # re_data["model_type"] = model_type
                # re_data["model_name"] = model_name
                
                if model_name not in ["SegFormer-Mit_b0-cityscapes","Segnext_Mscan-b_1-ade20k","Segnext_Mscan-b_1-cityscapes","Upernet_Convnext_base-ade20k"]:
                    continue
                    
                for div_type in ["class", "class_new", "pixel", "entropy", "fid", "is", "ssim", "lpips", "tce", "tie"]:
                    data = {}
                    for div_type2 in ["top", "random", "bottom"]: 
                        # 构建数据路径
                        if dataset == "cityscapes":
                            prefix_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes"
                            data_path = f"{prefix_path}/diversity_five_dim/diversity/{div_type}/{div_type2}/cityscapes/leftImg8bit/val"
                        else:
                            prefix_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/ADEChallengeData2016"
                            data_path = f"{prefix_path}/diversity_five_dim/diversity/{div_type}/{div_type2}/ADEChallengeData2016/images/validation"

                        # 检查路径是否存在
                        if not Path(data_path).exists():
                            print(f"    Warning: Path does not exist: {data_path}")
                            continue
                                
                        # 创建数据集，启用GPU预加载（如果数据集较小）
                        custom_dataset = CustomDataset(
                            data_path, 
                            preload=True if device == 'cuda' else False,
                            device=device
                        )
                        
                        if len(custom_dataset) == 0:
                            data[div_type2] = 0.0
                            continue
                        
                        # 计算SSIM多样性分数 - 现在自动使用CUDA加速
                        diversity_score = calculate_ssim_diversity_scores(custom_dataset, 0, batch_size=batch_size)
                        
                        data[div_type2] = diversity_score
                        
                        # 清理GPU内存
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                            
                                
                    print(f"  {model_type}, {model_name}, {div_type} bottom, top, random: ({data.get('bottom', 0):.2f},{data.get('top', 0):.2f},{data.get('random', 0):.2f})")
                    all_data.append(f"({data.get('bottom', 0):.2f},{data.get('top', 0):.2f},{data.get('random', 0):.2f})")
                    
    #             # 保存结果
    #             diver_path = f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-data-None/{dataset}/{model_type}/{model_name}/Select_{model_name}-diversity.csv"
                
    #             # 确保目录存在
    #             Path(diver_path).parent.mkdir(parents=True, exist_ok=True)
                
    #             # 读取或创建DataFrame
    #             if Path(diver_path).exists():
    #                 T = pd.read_csv(diver_path)
    #             else:
    #                 T = pd.DataFrame()
                
    #             # 更新SSIM列
    #             ssim_result = f"({data.get('bottom', 0):.2f},{data.get('top', 0):.2f},{data.get('random', 0):.2f})"
    #             T["SSIM"] = all_data
    #             T["dataset"] = [dataset]*len(T)
    #             T["model_type"] = [model_type]*len(T)
    #             T["model_name"] = [model_name]*len(T)
                
    #             T = pd.concat([T, re_data], axis=0)
                
    #             save_path = f"/home/ictt/xhr/code/DNNTesting/reSSNT/zzzzz/Select_{dataset}_{model_type}_{model_name}-diversity.csv"
    #             # 保存
    #             T.to_csv(save_path, index=False)

    #             all_data_save.append(T)
              

    # print("\nAll processing completed!")
    # torch.save(all_data_save, f"./all_data.pt")


    # # 最终清理
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    #     print(f"Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
