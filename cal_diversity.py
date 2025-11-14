from datetime import datetime
from mmengine.registry import init_default_scope
from mmseg.datasets import CityscapesDataset, ADE20KDataset
import random
import multiprocessing as mp
from image_diversity import ClipMetrics
from image_diversity import InceptionMetrics
from functools import partial
from pytorch_fid import fid_score
import lpips
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
import warnings
import logging
from functools import lru_cache
import gc

# Import from new modules
from datasets import get_img_names, CustomDataset, SSIMDataset
from diversity_metrics import *

try:
    from pytorch_msssim import ssim as torch_ssim
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    print("pytorch-msssim not found. Install with: pip install pytorch-msssim")
    HAS_PYTORCH_MSSSIM = False

# Set multiprocessing start method to spawn (CUDA compatible)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Ignore if already set

VALID_IMG_FORMATS = {"jpeg", "png", "gif", "bmp", "tiff"}

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning)
init_default_scope('mmseg')

def compute_ssim_scores(dataset: Dict, batch_size: int = 8):
    if torch.cuda.is_available():
        # Select batch size based on GPU memory
        fn = partial(calculate_ssim_diversity_scores_cuda, batch_size=batch_size, device='cuda')
    else:
        print("CUDA not available, using CPU version")
        fn = partial(calculate_ssim_diversity_scores_cpu_original, batch_size=batch_size)

    bottom_ssim_1 = fn(dataset["bottom"], idx=1)
    top_ssim_2 = fn(dataset["top"], idx=2)
    random_ssim_3 = fn(dataset["random"], idx=3)

    return bottom_ssim_1, top_ssim_2, random_ssim_3

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

@lru_cache(maxsize=128)
def get_dataset_config(dataset_name: str) -> Tuple[Dict, List]:
    """Cache dataset configuration"""
    
    if dataset_name == "cityscapes":
        data_prefix = dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val')
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(2048, 1024), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]
    else:
        data_prefix = dict(img_path='images/validation/', seg_map_path='annotations/validation/')
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(2048, 512), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]
        
    return data_prefix, pipeline

def pic_info_compute(func, data_root, save_path, data_prefix, pipeline):
    """Optimized image information computation"""
    results = []

    if "city" in data_root:
        dataset = CityscapesDataset(data_root=data_root, data_prefix=data_prefix, 
                                  test_mode=False, pipeline=pipeline)
    elif "ADE" in data_root:
        dataset = ADE20KDataset(data_root=data_root, data_prefix=data_prefix, 
                              test_mode=False, reduce_zero_label=True, pipeline=pipeline)

    results = compute_func(func, dataset)
    torch.save(results, save_path)
    return results

def compute_func(func, dataset):
    """Optimized parallel computation"""
    # Dynamically adjust process count
    num_processes = min(min(20, cpu_count()), len(dataset) // 10 + 1)
    results = []

    # Use chunksize to improve efficiency
    chunk_size = max(1, len(dataset) // (num_processes * 4))
    
    with Pool(processes=num_processes) as pool:
        results.extend(pool.map(func, dataset, chunksize=chunk_size))
    
    return results


def preprocess_images_batch(folder_path: str, save_path: str, batch_size: int = 32):
    """Batch preprocess images"""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    save_dir = Path(save_path) / "fid_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_img_names(folder_path)

    # Batch process images
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Preprocessing images"):
        batch_paths = image_paths[i:i + batch_size]
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                img = transforms.ToPILImage()(img)
                
                filename = Path(img_path).name
                save_file_path = save_dir / filename
                img.save(save_file_path)
            except Exception as e:
                print(f"Failed to process image {img_path}: {e}")
                continue

def compute_fid_scores(paths: str, batch_size: int = 100, num_workers: int = 8):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fid_value_tb = fid_score.calculate_fid_given_paths(
        [f"{paths}top/fid_test", f"{paths}bottom/fid_test"],
        batch_size=batch_size, device=device, dims=2048, num_workers=num_workers
    )
    fid_value_br = fid_score.calculate_fid_given_paths(
        [f"{paths}random/fid_test", f"{paths}bottom/fid_test"],
        batch_size=batch_size, device=device, dims=2048, num_workers=num_workers
    )
    fid_value_rt = fid_score.calculate_fid_given_paths(
        [f"{paths}random/fid_test", f"{paths}top/fid_test"],
        batch_size=batch_size, device=device, dims=2048, num_workers=num_workers
    )

    avg_fid_t = (fid_value_tb + fid_value_rt) / 2
    avg_fid_b = (fid_value_tb + fid_value_br) / 2
    avg_fid_r = (fid_value_br + fid_value_rt) / 2

    return avg_fid_b, avg_fid_t, avg_fid_r


def calculate_lpips_diversity_scores(dataset1: List, loss_fn, idx: int, batch_size: int = 16):
    """Vectorized LPIPS distance calculation"""
    distances = []
    device = next(loss_fn.parameters()).device
    
    progress = tqdm(total=len(dataset1) * len(dataset1), 
                   desc=f"   → Calculating {idx} LPIPS Distance", position=idx)
    
    # 批量计算LPIPS距离
    for i in range(0, len(dataset1), batch_size):
        batch1 = dataset1[i:i + batch_size]
        
        for j in range(0, len(dataset1), batch_size):
            batch2 = dataset1[j:j + batch_size]
            
            # 计算批内所有组合
            for img1 in batch1:
                img1_batch = img1.unsqueeze(0).to(device)
                for img2 in batch2:
                    img2_batch = img2.unsqueeze(0).to(device)
                    if torch.equal(img1_batch, img2_batch):
                        continue
                    try:
                        with torch.no_grad():
                            distance = loss_fn(img1_batch, img2_batch)
                            distances.append(distance.item())
                    except Exception as e:
                        print(f"LPIPS calculation error: {e}")
                        continue
                    finally:
                        progress.update()
    
    progress.close()
    return np.mean(distances) if distances else 0.0

def compute_lpips_scores(dataset: Dict, loss_fn, batch_size: int = 8):
    """Optimized LPIPS score calculation"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    loss_fn = loss_fn.to(device)
    
    # Sequential processing to avoid multiprocessing issues
    bottom_lpips_1 = calculate_lpips_diversity_scores(dataset["bottom"], loss_fn, 1, batch_size)
    top_lpips_2 = calculate_lpips_diversity_scores(dataset["top"], loss_fn, 2, batch_size)
    random_lpips_3 = calculate_lpips_diversity_scores(dataset["random"], loss_fn, 3, batch_size)

    return bottom_lpips_1, top_lpips_2, random_lpips_3

def calculate_inception_score(dataset, inception_model, batch_size: int = 100,
                                       splits: int = 10, device: str = 'cpu'):
    """Optimized Inception Score calculation"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)
    preds = []

    inception_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Inception Score"):
            batch = batch.to(device, non_blocking=True)
            pred = inception_model(batch)
            pred = torch.nn.functional.softmax(pred, dim=1)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Vectorized IS score calculation
    scores = []
    split_size = preds.shape[0] // splits
    
    for i in range(splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < splits - 1 else preds.shape[0]
        part = preds[start_idx:end_idx]
        
        py = np.mean(part, axis=0)
        kl_divergence = np.sum(part * (np.log(part + 1e-10) - np.log(py + 1e-10)), axis=1)
        scores.append(np.exp(np.mean(kl_divergence)))

    return np.mean(scores), np.std(scores)


def cal_diversity(dataset: str, dataset_path_prefix: str, model_name: str,
                           coverages_setting: Dict, N: int, device: str, save_path: str):
    """Optimized diversity calculation main function"""
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 初始化模型（只初始化一次）
    clip_metrics = ClipMetrics()
    inception_metrics = InceptionMetrics()

    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model.fc = nn.Identity()
    inception_model = inception_model.to(device)

    loss_fn = lpips.LPIPS(net='alex').to(device) 

    # 获取数据集配置
    data_prefix, pipeline = get_dataset_config(dataset)

    res = []
    prefix_path = f"{dataset_path_prefix}/diversity/{model_name}/select_files/"
    progress = tqdm(total=N, desc="   → Diversity Cal")
    
    for cov, hypers in coverages_setting.items():
        for hyper in hypers:
            datasets, ss_dataset, clas, pixs, ents, tces, ties, is_means = {}, {}, {}, {}, {}, {}, {}, {}
            
            for type_name in ["bottom", "top", "random"]:
                data_root = f"{prefix_path}/{cov}-{hyper}/{type_name}/"
                save_root = f"{dataset_path_prefix}/diversity/{model_name}/select_files/{cov}-{hyper}/{type_name}/"

                # Parallel compute diversity metrics
                clas[type_name] = only_cal_diversity_number("class", save_root, data_root, data_prefix, pipeline)
                pixs[type_name] = only_cal_diversity_number("pixel", save_root, data_root, data_prefix, pipeline)
                ents[type_name] = only_cal_diversity_number("entropy", save_root, data_root, data_prefix, pipeline)

                # Determine image path
                if dataset == "cityscapes":
                    img_path = f"{data_root}leftImg8bit/val"
                else:
                    img_path = f"{data_root}images/validation"

                # Calculate other metrics
                tces[type_name] = clip_metrics.tce(img_path, batch_size=100)
                ties[type_name] = inception_metrics.tie(img_path, batch_size=100)

                # Create optimized datasets
                datasets[type_name] = CustomDataset(img_path, preload=False)
                ss_dataset[type_name] = SSIMDataset(img_path, preload=False, device=device)

                # Calculate Inception Score
                is_means[type_name], _ = calculate_inception_score(
                    datasets[type_name], inception_model, device=device, batch_size=64
                )

                # Preprocess images
                preprocess_images_batch(img_path, f"{prefix_path}/{cov}-{hyper}/{type_name}/")

                # Clean up memory
                gc.collect()
                torch.cuda.empty_cache()

            # Calculate FID and LPIPS
            avg_fid_b, avg_fid_t, avg_fid_r = compute_fid_scores(
                f"{prefix_path}/{cov}-{hyper}/", batch_size=64, num_workers=6
            )
            avg_lpips_1, avg_lpips_2, avg_lpips_3 = compute_lpips_scores(
                datasets, loss_fn, batch_size=64
            )

            avg_ssim_1, avg_ssim_2, avg_ssim_3 = compute_ssim_scores(ss_dataset, batch_size=64)

            # Add results
            res.append([
                cov, hyper, 
                f"({clas['bottom'][0]},{clas['top'][0]},{clas['random'][0]})",
                f"({clas['bottom'][1]},{clas['top'][1]},{clas['random'][1]})",
                f"({pixs['bottom'][0]:.2f},{pixs['top'][0]:.2f},{pixs['random'][0]:.2f})",
                f"({pixs['bottom'][1]:.2f},{pixs['top'][1]:.2f},{pixs['random'][1]:.2f})",
                f"({ents['bottom'][0]:.2f},{ents['top'][0]:.2f},{ents['random'][0]:.2f})",
                f"({ents['bottom'][1]:.2f},{ents['top'][1]:.2f},{ents['random'][1]:.2f})",
                f"({tces['bottom']:.2f},{tces['top']:.2f},{tces['random']:.2f})",
                f"({ties['bottom']:.2f},{ties['top']:.2f},{ties['random']:.2f})",
                f"({is_means['bottom']:.2f},{is_means['top']:.2f},{is_means['random']:.2f})",
                f"({avg_fid_b:.2f},{avg_fid_t:.2f},{avg_fid_r:.2f})",
                f"({avg_lpips_1:.2f},{avg_lpips_2:.2f},{avg_lpips_3:.2f})",
                f"({avg_ssim_1:.2f},{avg_ssim_2:.2f},{avg_ssim_3:.2f})"
            ])
            
            gc.collect()
            torch.cuda.empty_cache()
            progress.update()

    # Save results
    result_df = pd.DataFrame(res, columns=[
        "cov", "hyper", "class", "class-ALL", "pixel", "pixel-ALL",
        "entropy", "entropy-Var", "TCE", "TIE", "IS", 'FID', 'LPIPS', 'SSIM'
    ])

    current_date = datetime.now().strftime("%Y-%m-%d")
    result_df.to_csv(f"{save_path}/Select_{model_name}-diversity.csv", index=False)
    
    return result_df


def cal_diversity_for_div(dataset: str, dataset_path_prefix: str, model_name: str,
                           device: str, save_path: str):
    """Optimized diversity calculation main function"""
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 初始化模型（只初始化一次）
    clip_metrics = ClipMetrics()
    inception_metrics = InceptionMetrics()

    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model.fc = nn.Identity()
    inception_model = inception_model.to(device)

    loss_fn = lpips.LPIPS(net='alex').to(device) 

    # 获取数据集配置
    data_prefix, pipeline = get_dataset_config(dataset)

    res = []
    prefix_path = f"{dataset_path_prefix}/diversity_five_dim/diversity/"
    progress = tqdm(total=5, desc="   → Diversity Cal")
    
    for div_type in ["class", "class_new", "pixel", "entropy", "fid", "is", "lpips", "tce", "tie", "ssim"]:
        datasets, ssimdataset, clas, pixs, ents, tces, ties, is_means = {}, {}, {}, {}, {}, {}, {}, {}
        for div_type2 in ["top", "random", "bottom"]:
            if dataset == "cityscapes":
                data_root = f"{prefix_path}/{div_type}/{div_type2}/cityscapes/"
            else:
                data_root = f"{prefix_path}/{div_type}/{div_type2}/ADEChallengeData2016/"
            
            save_root = f"{dataset_path_prefix}/diversity_five_dim/diversity/{model_name}/select_files/{div_type}/{div_type2}/"

            # 并行计算多样性指标
            clas[div_type2] = only_cal_diversity_number("class", save_root, data_root, data_prefix, pipeline)
            pixs[div_type2] = only_cal_diversity_number("pixel", save_root, data_root, data_prefix, pipeline)
            ents[div_type2] = only_cal_diversity_number("entropy", save_root, data_root, data_prefix, pipeline)
                
            # 确定图像路径
            if dataset == "cityscapes":
                img_path = f"{data_root}/leftImg8bit/val"
            else:
                img_path = f"{data_root}/images/validation"


            # Calculate other metrics
            tces[div_type2] = clip_metrics.tce(img_path, batch_size=100)
            ties[div_type2] = inception_metrics.tie(img_path, batch_size=100)

            # Create optimized datasets
            datasets[div_type2] = CustomDataset(img_path, preload=False)
            ssimdataset[div_type2] = SSIMDataset(img_path, preload=False, device=device)

            # Calculate Inception Score
            is_means[div_type2], _ = calculate_inception_score(
                datasets[div_type2], inception_model, device=device, batch_size=100
            )

            # Preprocess images
            preprocess_images_batch(img_path, f"{prefix_path}/{div_type}/{div_type2}/")

            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache()

        # Calculate FID and LPIPS
        avg_fid_b, avg_fid_t, avg_fid_r = compute_fid_scores(
            f"{prefix_path}/{div_type}/", batch_size=100, num_workers=6
        )
        avg_lpips_1, avg_lpips_2, avg_lpips_3 = compute_lpips_scores(
            datasets, loss_fn, batch_size=100
        )
        
        avg_ssim_1, avg_ssim_2, avg_ssim_3 = compute_ssim_scores(ssimdataset, batch_size=100)

        # 添加结果
        res.append([
            div_type, 
            f"({clas['bottom'][0]},{clas['top'][0]},{clas['random'][0]})",
            f"({clas['bottom'][1]},{clas['top'][1]},{clas['random'][1]})",
            f"({pixs['bottom'][0]:.2f},{pixs['top'][0]:.2f},{pixs['random'][0]:.2f})",
            f"({pixs['bottom'][1]:.2f},{pixs['top'][1]:.2f},{pixs['random'][1]:.2f})",
            f"({ents['bottom'][0]:.2f},{ents['top'][0]:.2f},{ents['random'][0]:.2f})",
            f"({ents['bottom'][1]:.2f},{ents['top'][1]:.2f},{ents['random'][1]:.2f})",
            f"({tces['bottom']:.2f},{tces['top']:.2f},{tces['random']:.2f})",
            f"({ties['bottom']:.2f},{ties['top']:.2f},{ties['random']:.2f})",
            f"({is_means['bottom']:.2f},{is_means['top']:.2f},{is_means['random']:.2f})",
            f"({avg_fid_b:.2f},{avg_fid_t:.2f},{avg_fid_r:.2f})",
            f"({avg_lpips_1:.2f},{avg_lpips_2:.2f},{avg_lpips_3:.2f})",
            f"({avg_ssim_1:.2f},{avg_ssim_2:.2f},{avg_ssim_3:.2f})"
        ])
        
        gc.collect()
        torch.cuda.empty_cache()
        progress.update()

    # 保存结果
    result_df = pd.DataFrame(res, columns=[
        "div_type", "class", "class-ALL", "pixel", "pixel-ALL",
        "entropy", "entropy-Var", "TCE", "TIE", "IS", 'FID', 'LPIPS', 'SSIM'
    ])

    result_df.to_csv(f"{save_path}/Select_{model_name}-diversity.csv", index=False)
    
    return result_df


def check_and_load_diversity(save_path: str, func, data):
    if os.path.exists(save_path):
        print(f"Loading existing SSIM data from {save_path}")
        data = torch.load(save_path)
        return data
    else:
        avg_ssim_1, avg_ssim_2, avg_ssim_3 = func(data, batch_size=100)
        ssim_dict = {"bottom": avg_ssim_2, "top": avg_ssim_1, "random": avg_ssim_3}
        torch.save(ssim_dict, save_path)
 
    gc.collect()
    torch.cuda.empty_cache()



def only_for_ssim(dataset: str, dataset_path_prefix: str, model_name: str,
                           coverages_setting: Dict, N: int, device: str, save_path: str):
    """Optimized diversity calculation main function"""
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_prefix, pipeline = get_dataset_config(dataset)

    res = []
    prefix_path = f"{dataset_path_prefix}/diversity/{model_name}/select_files/"
    
    pre_data_path = f"{save_path}/Select_{model_name}-diversity.csv"
    pre_df = pd.read_csv(pre_data_path)
    
    for cov, hypers in coverages_setting.items():
        for hyper in hypers:
            ss_dataset = {}
            ssim_path = f"{dataset_path_prefix}/diversity/{model_name}/select_files/{cov}-{hyper}/ssim.pth"
            if os.path.exists(ssim_path):
                print(f"Loading existing SSIM data from {ssim_path}")
                ssim_dict = torch.load(ssim_path)
                print(str(ssim_dict))
            else:
                for type_name in ["bottom", "top", "random"]:
                    data_root = f"{prefix_path}/{cov}-{hyper}/{type_name}/"
                    save_root = f"{dataset_path_prefix}/diversity/{model_name}/select_files/{cov}-{hyper}/{type_name}/"

                    if dataset == "cityscapes":
                        img_path = f"{data_root}leftImg8bit/val"
                    else:
                        img_path = f"{data_root}images/validation"

                    ss_dataset[type_name] = SSIMDataset(img_path, preload=False, device=device)
                        

                avg_ssim_1, avg_ssim_2, avg_ssim_3 = compute_ssim_scores(ss_dataset, batch_size=64)
                save_root = f"{dataset_path_prefix}/diversity/{model_name}/select_files/{cov}-{hyper}"
                ssim_dict = {"bottom": avg_ssim_2, "top": avg_ssim_1, "random": avg_ssim_3}
                torch.save(ssim_dict, f"{save_root}/ssim.pth")

            res.append([
                cov, hyper, 
                f"({ssim_dict['bottom']:.2f},{ssim_dict['top']:.2f},{ssim_dict['random']:.2f})"
            ])
            
            gc.collect()
            torch.cuda.empty_cache()

    result_df = pd.DataFrame(res, columns=[
        "cov", "hyper", 'SSIM'
    ])

    if result_df["cov"].equals(pre_df["cov"]):
        pre_df["SSIM"] = result_df["SSIM"]
        result_df = pre_df
    else:
        print("[cal_diversity.py  func: only_for_ssim] Error: hyper or cov do not match!")
        exit()

    result_df.to_csv(pre_data_path, index=False)
    
    return result_df


def cal_diversity_for_div_0921(dataset: str, dataset_path_prefix: str, model_name: str,
                           device: str, save_path: str):
    """Optimized diversity calculation main function"""

    # Initialize models (once only)
    clip_metrics = ClipMetrics()
    inception_metrics = InceptionMetrics()

    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model.fc = nn.Identity()
    inception_model = inception_model.to(device)

    loss_fn = lpips.LPIPS(net='alex').to(device) 

    data_prefix, pipeline = get_dataset_config(dataset)

    res = []
    prefix_path = f"{dataset_path_prefix}/diversity_five_dim/diversity/"
    progress = tqdm(total=5, desc="   → Diversity Cal")
    
    for div_type in ["class", "class_new", "pixel", "entropy", "fid", "is", "lpips", "tce", "tie", "ssim"]:
        datasets, ssimdataset, clas, pixs, ents, tces, ties, is_means = {}, {}, {}, {}, {}, {}, {}, {}
        for div_type2 in ["top", "random", "bottom"]:
            if dataset == "cityscapes":
                data_root = f"{prefix_path}/{div_type}/{div_type2}/cityscapes/"
            else:
                data_root = f"{prefix_path}/{div_type}/{div_type2}/ADEChallengeData2016/"
            
            save_root = f"{dataset_path_prefix}/diversity_five_dim/diversity/{model_name}/select_files/{div_type}/{div_type2}/"

            clas[div_type2] = only_cal_diversity_number("class", save_root, data_root, data_prefix, pipeline)
            pixs[div_type2] = only_cal_diversity_number("pixel", save_root, data_root, data_prefix, pipeline)
            ents[div_type2] = only_cal_diversity_number("entropy", save_root, data_root, data_prefix, pipeline)
                
            if dataset == "cityscapes":
                img_path = f"{data_root}/leftImg8bit/val"
            else:
                img_path = f"{data_root}/images/validation"

            tces[div_type2] = clip_metrics.tce(img_path, batch_size=100)
            ties[div_type2] = inception_metrics.tie(img_path, batch_size=100)

            datasets[div_type2] = CustomDataset(img_path, preload=False)
            ssimdataset[div_type2] = SSIMDataset(img_path, preload=False, device=device)

            is_means[div_type2], _ = calculate_inception_score(
                datasets[div_type2], inception_model, device=device, batch_size=100
            )

            preprocess_images_batch(img_path, f"{prefix_path}/{div_type}/{div_type2}/")

            gc.collect()
            torch.cuda.empty_cache()

        avg_fid_b, avg_fid_t, avg_fid_r = compute_fid_scores(
            f"{prefix_path}/{div_type}/", batch_size=100, num_workers=6
        )
        avg_lpips_1, avg_lpips_2, avg_lpips_3 = compute_lpips_scores(
            datasets, loss_fn, batch_size=100
        )
        
        avg_ssim_1, avg_ssim_2, avg_ssim_3 = compute_ssim_scores(ssimdataset, batch_size=100)

        res.append([
            div_type, 
            f"({clas['bottom'][0]},{clas['top'][0]},{clas['random'][0]})",
            f"({clas['bottom'][1]},{clas['top'][1]},{clas['random'][1]})",
            f"({pixs['bottom'][0]:.2f},{pixs['top'][0]:.2f},{pixs['random'][0]:.2f})",
            f"({pixs['bottom'][1]:.2f},{pixs['top'][1]:.2f},{pixs['random'][1]:.2f})",
            f"({ents['bottom'][0]:.2f},{ents['top'][0]:.2f},{ents['random'][0]:.2f})",
            f"({ents['bottom'][1]:.2f},{ents['top'][1]:.2f},{ents['random'][1]:.2f})",
            f"({tces['bottom']:.2f},{tces['top']:.2f},{tces['random']:.2f})",
            f"({ties['bottom']:.2f},{ties['top']:.2f},{ties['random']:.2f})",
            f"({is_means['bottom']:.2f},{is_means['top']:.2f},{is_means['random']:.2f})",
            f"({avg_fid_b:.2f},{avg_fid_t:.2f},{avg_fid_r:.2f})",
            f"({avg_lpips_1:.2f},{avg_lpips_2:.2f},{avg_lpips_3:.2f})",
            f"({avg_ssim_1:.2f},{avg_ssim_2:.2f},{avg_ssim_3:.2f})"
        ])
        
        gc.collect()
        torch.cuda.empty_cache()
        progress.update()

    result_df = pd.DataFrame(res, columns=[
        "div_type", "class", "class-ALL", "pixel", "pixel-ALL",
        "entropy", "entropy-Var", "TCE", "TIE", "IS", 'FID', 'LPIPS', 'SSIM'
    ])

    result_df.to_csv(f"{save_path}/Select_{model_name}-diversity.csv", index=False)
    
    return result_df