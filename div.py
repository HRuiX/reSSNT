import os
import json
import shutil
import tempfile
import lpips
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict
import warnings
from pytorch_fid import fid_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from skimage.metrics import structural_similarity as ssim

# Optional imports with fallbacks
try:
    from image_diversity import ClipMetrics, InceptionMetrics
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: image_diversity not available. TCE/TIE metrics will be disabled.")

warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    """统一的图像数据集类"""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, path
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            if self.transform:
                dummy = Image.new('RGB', (256, 256), (0, 0, 0))
                return self.transform(dummy), path
            return None, path


class ImprovedFIDContributionCalculator:
    """改进的FID贡献计算器 - 基于我们讨论的想法实现"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        
        
    
    def _safe_calculate_fid_fixed(self, path1: str, path2: str, batch_size: int) -> float:
        """
        快速修复版本的FID计算 - 解决尺寸不一致问题
        直接替换原代码中的 _safe_calculate_fid 方法
        """
        # 🔑 关键修复：预处理图像到统一尺寸
        processed_path1 = self._preprocess_images_to_uniform_size(path1, target_size=(299, 299))
        processed_path2 = self._preprocess_images_to_uniform_size(path2, target_size=(299, 299))
        
        try:
            # 使用预处理后的路径计算FID
            fid_score_val = fid_score.calculate_fid_given_paths(
                [processed_path1, processed_path2],
                batch_size=min(batch_size, 16),  # 限制batch size
                device=str(self.device),
                dims=2048,
                num_workers=0  # 禁用多进程
            )
            
            return fid_score_val
            
        except Exception as e:
            print(f"FID计算失败: {e}")
            raise e
        finally:
            # 清理临时目录
            if processed_path1 != path1:
                shutil.rmtree(processed_path1, ignore_errors=True)
            if processed_path2 != path2:
                shutil.rmtree(processed_path2, ignore_errors=True)
                
    
    def _preprocess_images_to_uniform_size(self, input_path: str, target_size=(299, 299)) -> str:
        """
        预处理图像到统一尺寸
        新增方法 - 添加到原来的类中
        """
        
        # 如果输入路径已经是临时目录，直接处理
        if "tmp" in input_path or "temp" in input_path:
            self._resize_images_in_place(input_path, target_size)
            return input_path
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix="fid_uniform_")
        
        # 获取所有图像文件
        image_paths = self._get_image_paths(input_path)
        
        print(f"预处理 {len(image_paths)} 张图像到尺寸 {target_size}")
        
        # 图像预处理变换
        transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        # 复制并调整所有图像尺寸
        copied_count = 0
        for img_path in image_paths:
            try:
                # 加载和调整图像
                with Image.open(img_path) as img:
                    # 确保是RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 调整尺寸
                    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
                    
                    # 保存到临时目录
                    filename = os.path.basename(img_path)
                    # 确保文件名是唯一的
                    base_name, ext = os.path.splitext(filename)
                    if not ext.lower() in ['.jpg', '.jpeg', '.png']:
                        ext = '.jpg'
                    
                    new_filename = f"{copied_count:06d}_{base_name}{ext}"
                    dst_path = os.path.join(temp_dir, new_filename)
                    
                    # 保存为JPEG格式以确保兼容性
                    if ext.lower() in ['.jpg', '.jpeg']:
                        img_resized.save(dst_path, 'JPEG', quality=95)
                    else:
                        img_resized.save(dst_path, 'PNG')
                    
                    copied_count += 1
                    
            except Exception as e:
                print(f"预处理图像失败 {os.path.basename(img_path)}: {e}")
                continue
        
        print(f"成功预处理 {copied_count} 张图像")
        
        if copied_count == 0:
            raise ValueError(f"没有成功预处理任何图像从 {input_path}")
        
        return temp_dir


    def _resize_images_in_place(self, directory: str, target_size=(299, 299)):
        """
        在目录中直接调整图像尺寸
        新增方法 - 添加到原来的类中
        """
        image_paths = self._get_image_paths(directory)
        
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 检查是否需要调整尺寸
                    if img.size != target_size:
                        img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
                        
                        # 保存回原文件
                        if img_path.lower().endswith(('.jpg', '.jpeg')):
                            img_resized.save(img_path, 'JPEG', quality=95)
                        else:
                            img_resized.save(img_path, 'PNG')
                            
            except Exception as e:
                print(f"调整图像尺寸失败 {os.path.basename(img_path)}: {e}")
                continue

    def _calculate_contributions_size_fixed(self, 
                                      image_paths_a: List[str],
                                      dataset_b_path: str,
                                      selected_indices: np.ndarray,
                                      baseline_fid: float,
                                      batch_size: int) -> np.ndarray:
        """
        修复版本的贡献计算 - 确保图像尺寸一致
        直接替换原代码中的 _calculate_contributions_standard 方法
        """
        print("\n计算单个图片FID贡献 (尺寸修复版)...")
        
        contributions = np.zeros(len(image_paths_a))
        temp_base_dir = tempfile.mkdtemp(prefix="fid_size_fixed_")
        
        try:
            for idx in tqdm(selected_indices, desc="计算贡献"):
                subset_dir = os.path.join(temp_base_dir, f"subset_{idx}")
                os.makedirs(subset_dir, exist_ok=True)
                
                # 复制其他图片并调整尺寸
                copied_count = 0
                target_size = (299, 299)
                
                for i, img_path in enumerate(image_paths_a):
                    if i != idx:
                        try:
                            # 🔑 关键修复：复制时就调整尺寸
                            with Image.open(img_path) as img:
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # 调整到统一尺寸
                                img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
                                
                                # 保存
                                dst_path = os.path.join(subset_dir, f"img_{i:06d}.jpg")
                                img_resized.save(dst_path, 'JPEG', quality=95)
                                copied_count += 1
                                
                        except Exception as e:
                            print(f"处理图像失败 {os.path.basename(img_path)}: {e}")
                            continue
                
                if copied_count < 2:
                    contributions[idx] = 0
                    shutil.rmtree(subset_dir, ignore_errors=True)
                    continue
                
                try:
                    # 🔑 使用修复版本的FID计算
                    subset_fid = self._safe_calculate_fid_fixed(
                        subset_dir, dataset_b_path, min(batch_size, 8)
                    )
                    contributions[idx] = subset_fid - baseline_fid
                except Exception as e:
                    print(f"FID计算失败，索引 {idx}: {e}")
                    contributions[idx] = 0
                
                shutil.rmtree(subset_dir, ignore_errors=True)
                
        finally:
            shutil.rmtree(temp_base_dir, ignore_errors=True)
        
        return contributions
    
    def calculate_image_fid_contributions(self, 
                                        dataset_a_path: str,
                                        dataset_b_path: str,
                                        batch_size: int = 50,
                                        sample_size: Optional[int] = None,
                                        use_efficient_method: bool = True) -> Dict:
        """
        计算数据集A中每张图片对FID的贡献
        返回格式修改为包含 path_to_score 字典
        """
        
        print("=" * 60)
        print("开始计算FID贡献分析")
        print("=" * 60)
        
        # 1. 获取数据集A中的所有图片路径
        image_paths_a = self._get_image_paths(dataset_a_path)
        n_images = len(image_paths_a)
        
        print(f"数据集A包含 {n_images} 张图片，路径为{dataset_a_path}")
        print(f"数据集B路径: {dataset_b_path}")
        
        if n_images == 0:
            raise ValueError("数据集A中没有找到图片")
            
        selected_indices = np.arange(n_images)
        print(f"分析所有 {n_images} 张图片")
        
        # 3. 计算基准FID（数据集A vs 数据集B）
        print("\n计算基准FID...")
        try:
            # baseline_fid = fid_score.calculate_fid_given_paths(
            #     [dataset_a_path, dataset_b_path],
            #     batch_size=batch_size,
            #     device=str(self.device),
            #     dims=2048,
            #     num_workers=0
            # )
            baseline_fid = self._safe_calculate_fid_fixed(
            dataset_a_path, dataset_b_path, min(batch_size, 16)
        )
            print(f"基准FID分数: {baseline_fid:.4f}")
        except Exception as e:
            print(f"计算基准FID失败: {e}")
            return {'error': str(e)}
        
        # 4. 计算每张图片的贡献
        # contributions = self._calculate_contributions_standard(
        #     image_paths_a, dataset_b_path, selected_indices, 
        #     baseline_fid, batch_size
        # )
        
        contributions = self._calculate_contributions_size_fixed(
            image_paths_a, dataset_b_path, selected_indices, 
            baseline_fid, min(batch_size, 16)
        )
        
        # 5. 创建路径到分数的字典
        path_to_score = {}
        for i, path in enumerate(image_paths_a):
            path_to_score[path] = float(contributions[i])
        
        # 6. 分析结果
        analysis = self._analyze_contributions(contributions, image_paths_a, selected_indices)
        
        return {
            'baseline_fid': baseline_fid,
            'contributions': contributions,
            'path_to_score': path_to_score,  # 新增：路径到分数的字典
            'analysis': analysis,
            'image_paths': [image_paths_a[i] for i in selected_indices],
            'selected_indices': selected_indices.tolist(),
            'dataset_info': {
                'dataset_a': dataset_a_path,
                'dataset_b': dataset_b_path,
                'total_images': n_images,
                'analyzed_images': len(selected_indices)
            }
        }
    
    def _get_image_paths(self, dataset_path: str) -> List[str]:
        """获取数据集中的所有图片路径"""
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_paths = []
        
        if os.path.isdir(dataset_path):
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            raise ValueError(f"数据集路径不存在或不是目录: {dataset_path}")
            
        return sorted(image_paths)
    
    def _calculate_contributions_efficient(self, 
                                         image_paths_a: List[str],
                                         dataset_b_path: str,
                                         selected_indices: np.ndarray,
                                         baseline_fid: float,
                                         batch_size: int) -> np.ndarray:
        print("\n使用高效方法计算FID贡献...")
        
        n_selected = len(selected_indices)
        contributions = np.zeros(len(image_paths_a))
        
        # 创建临时目录
        temp_base_dir = tempfile.mkdtemp(prefix="fid_analysis_")
        
        try:
            # 批量处理以提高效率
            batch_size_analysis = min(5, n_selected)  # 每批处理的图片数量
            
            for batch_start in tqdm(range(0, n_selected, batch_size_analysis), desc="批量计算FID贡献"):
                batch_end = min(batch_start + batch_size_analysis, n_selected)
                batch_indices = selected_indices[batch_start:batch_end]
                
                for idx in batch_indices:
                    # 创建不包含当前图片的子集
                    subset_dir = os.path.join(temp_base_dir, f"subset_{idx}")
                    os.makedirs(subset_dir, exist_ok=True)
                    
                    # 复制除当前图片外的所有图片 - 修复这里的逻辑错误
                    copied_count = 0
                    
                    for i, img_path in enumerate(image_paths_a):
                        if i != idx:  # 修复：正确排除当前要分析的图片
                            try:
                                dst_path = os.path.join(subset_dir, f"img_{i}_{os.path.basename(img_path)}")
                                shutil.copy2(img_path, dst_path)
                                copied_count += 1
                            except Exception as e:
                                print(f"复制图片失败 {img_path}: {e}")
                    
                    if copied_count < 2:
                        print(f"警告: 索引 {idx} 的子集图片数量不足 ({copied_count})")
                        contributions[idx] = 0
                        continue
                    
                    try:
                        # 计算移除该图片后的FID
                        subset_fid = fid_score.calculate_fid_given_paths(
                            [subset_dir, dataset_b_path],
                            batch_size=max(min(batch_size, copied_count), 2),
                            device=str(self.device),
                            dims=2048,
                            num_workers=1
                        )
                        
                        # 计算贡献: 移除图片后FID的变化
                        contributions[idx] = subset_fid - baseline_fid
                        
                    except Exception as e:
                        print(f"计算FID失败，索引 {idx}: {e}")
                        contributions[idx] = 0
                    
                    # 清理当前子集目录
                    shutil.rmtree(subset_dir, ignore_errors=True)
        finally:
            # 清理临时目录
            shutil.rmtree(temp_base_dir, ignore_errors=True)
        
        # 对未分析的图片使用插值估计
        analyzed_indices = set(selected_indices)
        if len(analyzed_indices) < len(image_paths_a):
            analyzed_contributions = contributions[selected_indices]
            mean_contribution = np.mean(analyzed_contributions)
            
            for i in range(len(image_paths_a)):
                if i not in analyzed_indices:
                    contributions[i] = mean_contribution
        
        return contributions
    
    def _calculate_contributions_standard(self, 
                                        image_paths_a: List[str],
                                        dataset_b_path: str,
                                        selected_indices: np.ndarray,
                                        baseline_fid: float,
                                        batch_size: int) -> np.ndarray:
        """标准的贡献计算方法 - 逐个处理"""
        print("\n使用标准方法计算FID贡献...")
        
        contributions = np.zeros(len(image_paths_a))
        temp_base_dir = tempfile.mkdtemp(prefix="fid_standard_")
        
        try:
            for idx in tqdm(selected_indices, desc="计算单个图片FID贡献"):
                subset_dir = os.path.join(temp_base_dir, f"subset_{idx}")
                os.makedirs(subset_dir, exist_ok=True)
                
                # 复制其他图片
                copied_count = 0
                for i, img_path in enumerate(image_paths_a):
                    if i != idx:
                        try:
                            dst_path = os.path.join(subset_dir, f"img_{i}_{os.path.basename(img_path)}")
                            shutil.copy2(img_path, dst_path)
                            copied_count += 1
                        except Exception as e:
                            print(f"复制失败 {img_path}: {e}")
                
                if copied_count < 2:
                    contributions[idx] = 0
                    shutil.rmtree(subset_dir, ignore_errors=True)
                    continue
                try:
                    subset_fid = fid_score.calculate_fid_given_paths(
                        [subset_dir, dataset_b_path],
                        batch_size=max(min(batch_size, copied_count), 2),
                        device=str(self.device),
                        dims=2048,
                        num_workers=1
                    )
                    contributions[idx] = subset_fid - baseline_fid
                except Exception as e:
                    print(f"FID计算失败，索引 {idx}: {e}")
                    contributions[idx] = 0
                shutil.rmtree(subset_dir, ignore_errors=True)
        finally:
            shutil.rmtree(temp_base_dir, ignore_errors=True)
        
        return contributions
    
    def _analyze_contributions(self, 
                             contributions: np.ndarray, 
                             image_paths: List[str], 
                             selected_indices: np.ndarray) -> Dict:
        """分析贡献结果"""
        
        analyzed_contributions = contributions[selected_indices]
        
        analysis = {
            'statistics': {
                'mean': float(np.mean(analyzed_contributions)),
                'std': float(np.std(analyzed_contributions)),
                'min': float(np.min(analyzed_contributions)),
                'max': float(np.max(analyzed_contributions)),
                'median': float(np.median(analyzed_contributions))
            },
            'quality_assessment': {
                'high_quality_count': int(np.sum(analyzed_contributions > 0)),
                'low_quality_count': int(np.sum(analyzed_contributions < 0)),
                'neutral_count': int(np.sum(analyzed_contributions == 0))
            }
        }
        
        # 识别最好和最差的图片
        sorted_indices = selected_indices[np.argsort(analyzed_contributions)]
        
        # 最差的图片（移除后FID显著改善）
        worst_indices = sorted_indices[:min(10, len(sorted_indices))]
        analysis['worst_images'] = {
            'indices': worst_indices.tolist(),
            'paths': [image_paths[i] for i in worst_indices],
            'contributions': [float(contributions[i]) for i in worst_indices]
        }
        
        # 最好的图片（移除后FID显著变差）
        best_indices = sorted_indices[-min(10, len(sorted_indices)):][::-1]
        analysis['best_images'] = {
            'indices': best_indices.tolist(),
            'paths': [image_paths[i] for i in best_indices],
            'contributions': [float(contributions[i]) for i in best_indices]
        }
        
        return analysis

    def save_results(self, results: Dict, output_dir: str):
        """保存FID贡献分析的详细结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整结果
        with open(os.path.join(output_dir, 'fid_detailed_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存分析摘要
        analysis = results.get('analysis', {})
        summary_lines = [
            "FID贡献分析摘要",
            "=" * 50,
            "",
            f"基准FID分数: {results.get('baseline_fid', 'N/A'):.4f}",
            "",
            "统计信息:",
            f"  平均贡献: {analysis.get('statistics', {}).get('mean', 0):.4f}",
            f"  标准差: {analysis.get('statistics', {}).get('std', 0):.4f}",
            f"  最小值: {analysis.get('statistics', {}).get('min', 0):.4f}",
            f"  最大值: {analysis.get('statistics', {}).get('max', 0):.4f}",
            f"  中位数: {analysis.get('statistics', {}).get('median', 0):.4f}",
            "",
            "质量评估:",
            f"  高质量图片数 (贡献>0): {analysis.get('quality_assessment', {}).get('high_quality_count', 0)}",
            f"  低质量图片数 (贡献<0): {analysis.get('quality_assessment', {}).get('low_quality_count', 0)}",
            f"  中性图片数 (贡献=0): {analysis.get('quality_assessment', {}).get('neutral_count', 0)}",
            "",
        ]
        
        # 添加最差图片信息
        worst_images = analysis.get('worst_images', {})
        if worst_images.get('paths'):
            summary_lines.extend([
                "最差图片 (移除后FID改善最多):",
                ""
            ])
            for i, (path, contrib) in enumerate(zip(worst_images['paths'], worst_images['contributions'])):
                summary_lines.append(f"  {i+1}. {os.path.basename(path)}: {contrib:.4f}")
            summary_lines.append("")
        
        # 添加最好图片信息
        best_images = analysis.get('best_images', {})
        if best_images.get('paths'):
            summary_lines.extend([
                "最好图片 (移除后FID变差最多):",
                ""
            ])
            for i, (path, contrib) in enumerate(zip(best_images['paths'], best_images['contributions'])):
                summary_lines.append(f"  {i+1}. {os.path.basename(path)}: {contrib:.4f}")
        
        with open(os.path.join(output_dir, 'fid_analysis_summary.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"FID详细分析结果已保存到: {output_dir}")


class ComprehensiveDiversityEvaluator:
    """综合多样性评估器 - 统一字典格式输出"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.results = {}
        self.path_to_score_results = {}  # 新增：统一的字典格式结果
        self.image_paths = []
        
        # 初始化改进的FID计算器
        self.fid_calculator = ImprovedFIDContributionCalculator(device)

        # 初始化各个评估器
        self._init_evaluators()

    def _init_evaluators(self):
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        self.inception_model.eval()
        self.inception_model = self.inception_model.to(self.device)

        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Initialize LPIPS
        try:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        except Exception as e:
            print(f"Warning: LPIPS not available: {e}")
            self.lpips_model = None

        # Initialize CLIP and Inception metrics if available
        if CLIP_AVAILABLE:
            try:
                self.clip_metrics = ClipMetrics()
                self.inception_metrics = InceptionMetrics()
            except Exception as e:
                print(f"Warning: Could not initialize CLIP/Inception metrics: {e}")
                self.clip_metrics = None
                self.inception_metrics = None
        else:
            self.clip_metrics = None
            self.inception_metrics = None

    def load_dataset(self, dataset_path: str,
                     extensions: Tuple[str] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        """加载数据集"""
        self.image_path = dataset_path
        self.image_paths = []

        if os.path.isdir(dataset_path):
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(extensions):
                        self.image_paths.append(os.path.join(root, file))
        else:
            raise ValueError("数据集路径必须是一个文件夹")

        print(f"发现 {len(self.image_paths)} 张图片")
        return len(self.image_paths)

    def _create_path_to_score_dict(self, scores: np.ndarray, valid_paths: List[str] = None) -> Dict[str, float]:
        """创建路径到分数的字典"""
        if valid_paths is None:
            valid_paths = self.image_paths
        
        if len(scores) != len(valid_paths):
            # 如果分数数量与路径不匹配，尝试对齐
            min_len = min(len(scores), len(valid_paths))
            scores = scores[:min_len]
            valid_paths = valid_paths[:min_len]
        
        return {path: float(score) for path, score in zip(valid_paths, scores)}

    def compute_fid_contributions_improved(self, 
                                         reference_dataset_path: str,
                                         batch_size: int = 50,
                                         sample_size: Optional[int] = None) -> Dict[str, float]:
        """使用改进的FID贡献计算方法，返回字典格式"""
        
        # 使用当前数据集路径
        current_dataset_path = self.image_path
        if not current_dataset_path:
            raise ValueError("请先加载数据集")
        
        # 调用改进的FID计算器
        results = self.fid_calculator.calculate_image_fid_contributions(
            dataset_a_path=current_dataset_path,
            dataset_b_path=reference_dataset_path,
            batch_size=batch_size,
            sample_size=sample_size,
            use_efficient_method=True
        )
        
        if 'error' in results:
            raise RuntimeError(f"FID计算失败: {results['error']}")
        
        # 存储详细结果用于后续分析
        self.fid_detailed_results = results
        
        return results['path_to_score']

    def compute_is_contributions(self, batch_size: int = 32) -> Dict[str, float]:
        """计算每张图片对IS的贡献，返回字典格式"""
        print("计算IS贡献...")

        # 提取所有图片的预测分布
        dataset = ImageDataset(self.image_paths, self.inception_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        predictions = []
        valid_paths = []

        self.inception_model.eval()
        with torch.no_grad():
            for batch_images, batch_paths in tqdm(dataloader, desc="提取Inception特征"):
                # Skip None leftImg8bit
                valid_batch = []
                valid_batch_paths = []
                
                for img, path in zip(batch_images, batch_paths):
                    if img is not None:
                        valid_batch.append(img)
                        valid_batch_paths.append(path)
                
                if not valid_batch:
                    continue
                    
                batch_tensor = torch.stack(valid_batch).to(self.device)
                outputs = self.inception_model(batch_tensor)
                probs = F.softmax(outputs, dim=1)

                predictions.append(probs.cpu().numpy())
                valid_paths.extend(valid_batch_paths)

        if not predictions:
            return {}
            
        predictions = np.concatenate(predictions, axis=0)

        # 使用简化的Shapley值近似方法
        shapley_values = self._compute_shapley_values_fast(predictions)

        return self._create_path_to_score_dict(shapley_values, valid_paths)

    def _compute_shapley_values_fast(self, predictions: np.ndarray, sample_size: int = 100) -> np.ndarray:
        """快速近似计算Shapley值 - 减少采样以提高速度"""
        n = len(predictions)
        shapley_values = np.zeros(n)

        # Reduce sample size for faster computation
        # effective_sample_size = min(sample_size, max(10, n // 10))
        effective_sample_size = n

        for i in tqdm(range(n), desc=f"计算Shapley值，个数为{len(predictions)}"):
            marginal_contributions = []
            other_indices = [j for j in range(n) if j != i]

            for _ in range(effective_sample_size):
                # 随机选择子集大小
                max_subset_size = min(len(other_indices), 20)  # Limit subset size
                subset_size = np.random.randint(0, max_subset_size + 1)

                if subset_size == 0:
                    subset = []
                else:
                    subset = np.random.choice(other_indices,
                                              size=subset_size,
                                              replace=False).tolist()

                # 计算边际贡献
                is_without = self._compute_is_score(predictions[subset]) if subset else 1.0
                subset_with_i = subset + [i]
                is_with = self._compute_is_score(predictions[subset_with_i])

                marginal_contribution = is_with - is_without
                marginal_contributions.append(marginal_contribution)

            shapley_values[i] = np.mean(marginal_contributions)

        return shapley_values

    def _compute_is_score(self, predictions: np.ndarray) -> float:
        """计算IS分数"""
        if len(predictions) == 0:
            return 1.0

        # 计算边际分布
        marginal_dist = np.mean(predictions, axis=0)

        # 计算KL散度
        kl_divergences = []
        for pred in predictions:
            kl_div = np.sum(pred * (np.log(pred + 1e-10) -
                                    np.log(marginal_dist + 1e-10)))
            kl_divergences.append(kl_div)

        return np.exp(np.mean(kl_divergences))

    def compute_lpips_diversity_scores(self, batch_size: int = 50, 
                                     sample_size: Optional[int] = None) -> Dict[str, float]:
        """计算LPIPS多样性分数，返回字典格式"""
        if self.lpips_model is None:
            raise RuntimeError("LPIPS not available. Please install lpips.")

        print("计算LPIPS多样性分数...")
        n_images = len(self.image_paths)
        
        # If dataset is large, use sampling
        if sample_size and sample_size < n_images:
            indices = np.random.choice(n_images, sample_size, replace=False)
            sampled_paths = [self.image_paths[i] for i in indices]
            n_images = sample_size
            image_paths = sampled_paths
        else:
            image_paths = self.image_paths
            
        # 预加载所有图片
        images = []
        valid_paths = []
        for img_path in tqdm(image_paths, desc="加载图片"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.lpips_transform(img).unsqueeze(0).to(self.device)
                images.append(img_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"加载图片失败 {img_path}: {e}")

        # 计算每张图片的多样性分数
        diversity_scores = []
        for i in tqdm(range(len(images)), desc="计算LPIPS多样性"):
            distances = []
            for j in range(len(images)):
                if i != j:
                    try:
                        with torch.no_grad():
                            lpips_dist = self.lpips_model(images[i], images[j]).item()
                        distances.append(lpips_dist)
                    except Exception as e:
                        print(f"计算LPIPS失败 {i}-{j}: {e}")
            
            diversity_scores.append(np.mean(distances) if distances else 0.0)

        return self._create_path_to_score_dict(np.array(diversity_scores), valid_paths)
    def compute_ssim_diversity_scores(self, batch_size: int = 50, 
                                sample_size: Optional[int] = None) -> Dict[str, float]:
        """计算SSIM多样性分数，返回字典格式"""
        print("计算SSIM多样性分数...")
        n_images = len(self.image_paths)
        
        image_paths = self.image_paths
            
        # 预加载所有图片（转换为numpy数组）
        images = []
        valid_paths = []
        
        # SSIM专用的transform（不需要归一化）
        ssim_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        for img_path in tqdm(image_paths, desc="加载图片"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = ssim_transform(img)
                # 转换为numpy格式 (H, W, C)，范围[0,1]
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                images.append(img_np)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"加载图片失败 {img_path}: {e}")

        # 计算每张图片的SSIM多样性分数
        diversity_scores = []
        for i in tqdm(range(len(images)), desc="计算SSIM多样性"):
            ssim_values = []
            for j in range(len(images)):
                if i != j:
                    try:
                        # 计算RGB图像的SSIM
                        ssim_val = ssim(images[i], images[j], data_range=1.0, 
                                    channel_axis=2, multichannel=True)
                        ssim_values.append(ssim_val)
                    except Exception as e:
                        print(f"计算SSIM失败 {i}-{j}: {e}")
            
            # 计算平均SSIM，然后转换为距离度量（1-SSIM）以保持与其他指标的一致性
            if ssim_values:
                avg_ssim = np.mean(ssim_values)
                diversity_score = 1.0 - avg_ssim  # 转换为距离度量，值越大表示越不相似（越多样）
            else:
                diversity_score = 0.0
                
            diversity_scores.append(diversity_score)

        return self._create_path_to_score_dict(np.array(diversity_scores), valid_paths)


    def compute_tce_contributions(self, batch_size: int = 30) -> Dict[str, float]:
        """计算TCE贡献，返回字典格式"""
        if self.clip_metrics is None:
            print("TCE not available - CLIP metrics not initialized")
            return {}

        print("计算TCE贡献...")
        try:
            dataset_dir = self.image_path
            contributions, full_tce = self.clip_metrics.calculate_individual_tce_contributions(
                dataset_dir, batch_size=batch_size
            )

            return contributions
        except Exception as e:
            print(f"TCE计算失败: {e}")
            return {}

    def compute_tie_contributions(self, batch_size: int = 30) -> Dict[str, float]:
        """计算TIE贡献，返回字典格式"""
        if self.inception_metrics is None:
            print("TIE not available - Inception metrics not initialized")
            return {}

        print("计算TIE贡献...")
        try:
            dataset_dir = self.image_path
            contributions, full_tie = self.inception_metrics.calculate_individual_tie_contributions(
                dataset_dir, batch_size=batch_size
            )

            return contributions
        except Exception as e:
            print(f"TIE计算失败: {e}")
            return {}

    def evaluate_all_metrics(self,
                             reference_dataset_path: Optional[str] = None,
                             compute_tce: bool = True,
                             compute_tie: bool = True,
                             compute_is: bool = True,
                             compute_fid: bool = True,
                             compute_lpips: bool = True,
                             compute_ssim: bool = True,
                             lpips_sample_size: Optional[int] = None,
                             ssim_sample_size: Optional[int] = None,
                             fid_sample_size: int = 500) -> Dict[str, Dict[str, float]]:
        """计算所有指标，统一返回字典格式"""
        
        print("\n" + "=" * 60)
        print("开始计算所有多样性指标")
        print("=" * 60)
        
        # # TCE
        # if compute_tce and self.clip_metrics is not None:
        #     try:
        #         tce_dict = self.compute_tce_contributions()
        #         if tce_dict:
        #             self.path_to_score_results['tce'] = tce_dict
        #             scores = list(tce_dict.values())
        #             print(f"TCE计算完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        #         else:
        #             self.path_to_score_results['tce'] = {}
        #     except Exception as e:
        #         print(f"TCE计算失败: {e}")
        #         self.path_to_score_results['tce'] = {}

        # # TIE
        # if compute_tie and self.inception_metrics is not None:
        #     try:
        #         tie_dict = self.compute_tie_contributions()
        #         if tie_dict:
        #             self.path_to_score_results['tie'] = tie_dict
        #             scores = list(tie_dict.values())
        #             print(f"TIE计算完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        #         else:
        #             self.path_to_score_results['tie'] = {}
        #     except Exception as e:
        #         print(f"TIE计算失败: {e}")
        #         self.path_to_score_results['tie'] = {}

        # # IS
        # if compute_is:
        #     try:
        #         is_dict = self.compute_is_contributions()
        #         if is_dict:
        #             self.path_to_score_results['is'] = is_dict
        #             scores = list(is_dict.values())
        #             print(f"IS计算完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        #         else:
        #             self.path_to_score_results['is'] = {}
        #     except Exception as e:
        #         print(f"IS计算失败: {e}")
        #         self.path_to_score_results['is'] = {}

        # # FID
        # if compute_fid and reference_dataset_path:
        #     try:
        #         fid_dict = self.compute_fid_contributions_improved(
        #             reference_dataset_path, 
        #             sample_size=fid_sample_size,
        #         )
        #         if fid_dict:
        #             self.path_to_score_results['fid'] = fid_dict
        #             scores = list(fid_dict.values())
        #             print(f"FID计算完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        #         else:
        #             self.path_to_score_results['fid'] = {}
        #     except Exception as e:
        #         print(f"FID计算失败: {e}")
        #         self.path_to_score_results['fid'] = {}

        # # LPIPS
        # if compute_lpips and self.lpips_model is not None:
        #     try:
        #         lpips_dict = self.compute_lpips_diversity_scores(sample_size=lpips_sample_size)
        #         if lpips_dict:
        #             self.path_to_score_results['lpips'] = lpips_dict
        #             scores = list(lpips_dict.values())
        #             print(f"LPIPS计算完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        #         else:
        #             self.path_to_score_results['lpips'] = {}
        #     except Exception as e:
        #         print(f"LPIPS计算失败: {e}")
        #         self.path_to_score_results['lpips'] = {}

        # SSIM
        if compute_ssim:
            try:
                ssim_dict = self.compute_ssim_diversity_scores(sample_size=ssim_sample_size)
                if ssim_dict:
                    self.path_to_score_results['ssim'] = ssim_dict
                    scores = list(ssim_dict.values())
                    print(f"SSIM计算完成: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
                else:
                    self.path_to_score_results['ssim'] = {}
            except Exception as e:
                print(f"SSIM计算失败: {e}")
                self.path_to_score_results['ssim'] = {}

        return self.path_to_score_results

    def select_diverse_images_by_score(self, metric: str, n_select: int = 100,
                                     most_diverse: bool = True) -> Tuple[List[str], List[float]]:
        """根据指定指标选择最多样或最不多样的图片，返回路径和分数"""
        if metric not in self.path_to_score_results or not self.path_to_score_results[metric]:
            raise ValueError(f"指标 {metric} 的结果不可用")

        score_dict = self.path_to_score_results[metric]
        
        # 按分数排序
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=most_diverse)
        
        # 选择前n_select个
        selected_items = sorted_items[:n_select]
        selected_paths = [item[0] for item in selected_items]
        selected_scores = [item[1] for item in selected_items]

        return selected_paths, selected_scores

    def save_path_to_score_results(self, output_dir: str):
        """保存所有指标的{路径：分数}字典格式结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存每个指标的字典
        for metric_name, score_dict in self.path_to_score_results.items():
            if score_dict:
                output_file = os.path.join(output_dir, f'{metric_name}_path_to_score.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(score_dict, f, indent=2, ensure_ascii=False)
                print(f"已保存 {metric_name} 结果到: {output_file}")

        # 保存所有指标的汇总文件
        summary_file = os.path.join(output_dir, 'all_metrics_path_to_score.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.path_to_score_results, f, indent=2, ensure_ascii=False)
        print(f"已保存所有指标汇总到: {summary_file}")

        # 保存详细的FID分析结果（如果存在）
        if hasattr(self, 'fid_detailed_results') and self.fid_detailed_results is not None:
            self.fid_calculator.save_results(
                self.fid_detailed_results, 
                os.path.join(output_dir, 'fid_detailed_analysis')
            )

        # 生成选择结果报告
        self._generate_selection_report(output_dir)

    def _generate_selection_report(self, output_dir: str):
        """生成每个指标的选择结果报告"""
        report_lines = [
            "图像多样性评估选择报告",
            "=" * 50,
            ""
        ]

        for metric in self.path_to_score_results:
            if not self.path_to_score_results[metric]:
                continue
                
            try:
                # 最多样的10张
                most_diverse_paths, most_diverse_scores = self.select_diverse_images_by_score(
                    metric, n_select=10, most_diverse=True
                )
                
                # 最不多样的10张
                least_diverse_paths, least_diverse_scores = self.select_diverse_images_by_score(
                    metric, n_select=10, most_diverse=False
                )

                report_lines.extend([
                    f"{metric.upper()} 指标分析:",
                    f"  总图片数: {len(self.path_to_score_results[metric])}",
                    f"  分数范围: {min(self.path_to_score_results[metric].values()):.4f} ~ {max(self.path_to_score_results[metric].values()):.4f}",
                    "",
                    f"  最多样的10张图片:",
                ])
                
                for i, (path, score) in enumerate(zip(most_diverse_paths, most_diverse_scores)):
                    report_lines.append(f"    {i+1}. {os.path.basename(path)}: {score:.4f}")
                
                report_lines.extend([
                    "",
                    f"  最不多样的10张图片:",
                ])
                
                for i, (path, score) in enumerate(zip(least_diverse_paths, least_diverse_scores)):
                    report_lines.append(f"    {i+1}. {os.path.basename(path)}: {score:.4f}")
                
                report_lines.extend(["", ""])
                
            except Exception as e:
                report_lines.append(f"  {metric.upper()} 分析出错: {e}")
                report_lines.append("")

        # 保存报告
        with open(os.path.join(output_dir, 'selection_report.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

    def visualize_score_distributions(self, output_dir: str):
        """可视化所有指标的分数分布"""
        os.makedirs(output_dir, exist_ok=True)

        if not self.path_to_score_results:
            print("没有可用的数据进行可视化")
            return

        # 过滤出有效的指标
        valid_metrics = {k: v for k, v in self.path_to_score_results.items() if v}
        
        if not valid_metrics:
            print("没有有效的指标数据")
            return

        n_metrics = len(valid_metrics)
        fig, axes = plt.subplots(2, n_metrics, figsize=(5 * n_metrics, 10))

        if n_metrics == 1:
            axes = axes.reshape(-1, 1)

        for idx, (metric_name, score_dict) in enumerate(valid_metrics.items()):
            scores = list(score_dict.values())
            
            # 直方图
            ax = axes[0, idx]
            ax.hist(scores, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric_name.upper()} 分数分布')
            ax.set_xlabel('分数')
            ax.set_ylabel('频次')

            # 添加统计信息
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.5)
            ax.legend()

            # 箱线图
            ax = axes[1, idx]
            ax.boxplot(scores)
            ax.set_title(f'{metric_name.upper()} 箱线图')
            ax.set_ylabel('分数')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"分数分布可视化已保存到: {output_dir}")


def main():
    """主函数示例"""
    # 配置参数
    # dataset_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/ADEChallengeData2016/leftImg8bit/validation"  # 请替换为实际路径
    dataset_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes/leftImg8bit/val"  # 请替换为实际路径
    reference_dataset = dataset_path  # FID计算通常使用同一数据集作为参考
    output_dir = "./diversity_results_dict_format"

    # 创建评估器
    evaluator = ComprehensiveDiversityEvaluator()

    # 加载数据集
    n_images = evaluator.load_dataset(dataset_path)
    print(f"加载了 {n_images} 张图片")

    # 评估所有指标
    print("\n" + "=" * 60)
    print("开始计算所有多样性指标 (统一字典格式)")
    print("=" * 60)

    path_to_score_results = evaluator.evaluate_all_metrics(
        reference_dataset_path=reference_dataset,
        compute_tce=True,  # 如果没有image_diversity库则设为False
        compute_tie=True,  # 如果没有image_diversity库则设为False
        compute_is=True,
        compute_fid=True,
        compute_lpips=True,
        lpips_sample_size=n_images,  # 采样500张图片计算LPIPS
        ssim_sample_size=n_images,   # 新增：SSIM采样大小
        fid_sample_size=n_images,     # 采样500张图片计算FID贡献
    )

    # 保存字典格式结果
    evaluator.save_path_to_score_results(output_dir)
    
    # 可视化结果
    evaluator.visualize_score_distributions(output_dir)

    # 输出摘要
    print("\n" + "=" * 60)
    print("计算完成摘要")
    print("=" * 60)
    
    for metric_name, score_dict in path_to_score_results.items():
        if score_dict:
            scores = list(score_dict.values())
            print(f"{metric_name.upper()}: 图片数={len(scores)}, 均值={np.mean(scores):.4f}, 标准差={np.std(scores):.4f}")

    print(f"\n完成！所有结果以{{路径:分数}}字典格式保存在 {output_dir}")
    print("\n保存的文件包括:")
    print("- *_path_to_score.json: 各指标的{路径:分数}字典")
    print("- all_metrics_path_to_score.json: 所有指标的汇总字典")
    print("- selection_report.txt: 各指标的最佳/最差图片选择报告")
    print("- score_distributions.png: 分数分布可视化")


if __name__ == "__main__":
    main()