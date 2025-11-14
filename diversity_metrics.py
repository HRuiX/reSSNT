import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms
from typing import List
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool, cpu_count

try:
    from pytorch_msssim import ssim as torch_ssim

    HAS_PYTORCH_MSSSIM = True
except ImportError:
    print("Warning: pytorch-msssim not found. Install with: pip install pytorch-msssim")
    HAS_PYTORCH_MSSSIM = False


def rgb_to_grayscale_gpu(image_batch):
    """
    Convert RGB images to grayscale on GPU.

    Args:
        image_batch: Batch of RGB images (batch_size, 3, height, width)

    Returns:
        Grayscale images (batch_size, 1, height, width)
    """
    # Use standard RGB to grayscale conversion weights
    weights = torch.tensor([0.299, 0.587, 0.114], device=image_batch.device).view(
        1, 3, 1, 1
    )
    return torch.sum(image_batch * weights, dim=1, keepdim=True)


def calculate_ssim_gpu_batch(img1_batch, img2_batch):
    """
    Calculate SSIM using GPU in batch mode.

    Args:
        img1_batch: First batch of images (batch_size, channels, height, width)
        img2_batch: Second batch of images (batch_size, channels, height, width)

    Returns:
        SSIM values for each pair
    """
    if HAS_PYTORCH_MSSSIM:
        # Use pytorch-msssim with GPU acceleration
        ssim_values = torch_ssim(
            img1_batch, img2_batch, data_range=1.0, size_average=False
        )
        return ssim_values
    else:
        # Fall back to manual GPU SSIM implementation
        return manual_ssim_gpu(img1_batch, img2_batch)


def manual_ssim_gpu(img1_batch, img2_batch, window_size=11, sigma=1.5):
    """
    Manual GPU implementation of SSIM.

    Args:
        img1_batch: First batch of images (batch_size, channels, height, width)
        img2_batch: Second batch of images (batch_size, channels, height, width)
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian window

    Returns:
        SSIM values for each pair
    """
    device = img1_batch.device

    # Convert to grayscale if RGB
    if img1_batch.shape[1] == 3:
        img1_batch = rgb_to_grayscale_gpu(img1_batch)
        img2_batch = rgb_to_grayscale_gpu(img2_batch)

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)

    # SSIM constants
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    # Calculate means
    mu1 = F.conv2d(img1_batch, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2_batch, window, padding=window_size // 2, groups=1)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Calculate variance and covariance
    sigma1_sq = (
        F.conv2d(img1_batch**2, window, padding=window_size // 2, groups=1) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2_batch**2, window, padding=window_size // 2, groups=1) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1_batch * img2_batch, window, padding=window_size // 2, groups=1)
        - mu1_mu2
    )

    # SSIM calculation
    numerator1 = 2 * mu1_mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)

    # Return average SSIM value for each image
    return ssim_map.mean(dim=[1, 2, 3])


def calculate_ssim_diversity_scores_cuda(
    dataset1: List, idx: int, batch_size: int = 16, device: str = "cuda"
):
    """
    Calculate SSIM diversity scores using CUDA acceleration.

    Args:
        dataset1: List of images to compare
        idx: Index for progress bar positioning
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Diversity score (1.0 - mean_ssim)
    """
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    device = torch.device(device)

    # Preprocess and move to GPU
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

    # Stack all images
    all_images = torch.stack(gpu_images)  # (N, C, H, W)
    del gpu_images  # Free memory

    ssim_values = []
    n_images = all_images.shape[0]

    progress = tqdm(
        total=n_images * (n_images - 1),
        desc=f"Calculating SSIM on {device}",
        leave=True,
    )

    # Process in chunks to save memory
    chunk_size = min(batch_size, n_images)

    for i in range(0, n_images, chunk_size):
        chunk1 = all_images[i : i + chunk_size]  # (chunk_size, C, H, W)

        for j in range(0, n_images, chunk_size):
            chunk2 = all_images[j : j + chunk_size]  # (chunk_size, C, H, W)

            # Expand dimensions for batch comparison
            chunk1_expanded = chunk1.unsqueeze(1)  # (chunk_size, 1, C, H, W)
            chunk2_expanded = chunk2.unsqueeze(0)  # (1, chunk_size, C, H, W)

            # Calculate SSIM for all combinations
            for idx1 in range(chunk1.shape[0]):
                for idx2 in range(chunk2.shape[0]):
                    actual_i = i + idx1
                    actual_j = j + idx2

                    if actual_i == actual_j:
                        continue

                    img1 = chunk1[idx1 : idx1 + 1]  # (1, C, H, W)
                    img2 = chunk2[idx2 : idx2 + 1]  # (1, C, H, W)

                    try:
                        ssim_val = calculate_ssim_gpu_batch(img1, img2)
                        ssim_values.append(ssim_val.item())
                    except Exception as e:
                        print(f"SSIM calculation error: {e}")

                    progress.update(1)

    progress.close()

    # Clean up GPU memory
    del all_images
    torch.cuda.empty_cache()

    mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
    return 1.0 - mean_ssim


def calculate_ssim_diversity_scores_cpu_original(
    dataset1: List, idx: int, batch_size: int = 16
):
    """
    Original CPU version as backup.

    Args:
        dataset1: List of images to compare
        idx: Index for progress bar positioning
        batch_size: Batch size for processing

    Returns:
        Diversity score (1.0 - mean_ssim)
    """
    from skimage.metrics import structural_similarity as ssim

    ssim_values = []

    progress = tqdm(
        total=len(dataset1) * (len(dataset1) - 1),
        desc=f"   → Calculating {idx} SSIM Distance (CPU)",
        position=idx,
    )

    for i in range(0, len(dataset1), batch_size):
        batch1 = dataset1[i : i + batch_size]

        for j in range(0, len(dataset1), batch_size):
            batch2 = dataset1[j : j + batch_size]

            for img1 in batch1:
                img1_np = img1.permute(1, 2, 0).numpy()

                for img2 in batch2:
                    if torch.equal(img1, img2):
                        continue
                    try:
                        img2_np = img2.permute(1, 2, 0).numpy()
                        ssim_val = ssim(
                            img1_np,
                            img2_np,
                            data_range=1.0,
                            channel_axis=2,
                            multichannel=True,
                        )
                        ssim_values.append(ssim_val)

                    except Exception as e:
                        print(f"SSIM calculation error: {e}")
                        continue
                    finally:
                        progress.update()

    progress.close()
    mean_ssim = np.mean(ssim_values) if ssim_values else 0.0
    return 1.0 - mean_ssim


def calculate_ssim_diversity_scores(
    dataset, idx: int, batch_size: int = 16, device: str = "cuda"
):
    """
    Wrapper function to calculate SSIM diversity scores.

    Automatically selects GPU or CPU version based on availability.

    Args:
        dataset: List of images to compare
        idx: Index for progress bar positioning
        batch_size: Batch size for processing
        device: Preferred device ('cuda' or 'cpu')

    Returns:
        Diversity score (1.0 - mean_ssim)
    """
    if torch.cuda.is_available() and device == "cuda":
        return calculate_ssim_diversity_scores_cuda(dataset, idx, batch_size, device)
    else:
        return calculate_ssim_diversity_scores_cpu_original(dataset, idx, batch_size)


def compute_pic_class(data):
    """
    Calculate image class diversity.

    Computes the number of unique semantic classes in a segmentation map.

    Args:
        data: Dictionary containing segmentation data with 'data_samples' key

    Returns:
        List containing [file_path, num_unique_pixels, unique_class_values]
    """
    file_path = data["data_samples"].seg_map_path
    image_tensor = data["data_samples"].gt_sem_seg.data
    image_tensor = image_tensor.view(-1, image_tensor.shape[-1])
    unique_class_values = torch.unique(image_tensor)
    num_unique_pixels = unique_class_values.shape[0]
    unique_class_values = [str(i) for i in unique_class_values.tolist()]
    return [file_path, num_unique_pixels, unique_class_values]


def compute_pic_pixel(data):
    """
    Calculate image pixel diversity.

    Computes the number of unique RGB pixel values in an image.

    Args:
        data: Dictionary containing image data with 'inputs' and 'data_samples' keys

    Returns:
        List containing [file_path, num_unique_pixels, unique_pixels]
    """
    file_path = data["data_samples"].img_path
    tensor = data["inputs"]
    tensor_reshaped = tensor.permute(1, 2, 0).numpy()
    unique_pixels = np.unique(tensor_reshaped.reshape(-1, 3), axis=0)
    num_unique_pixels = len(unique_pixels)
    unique_pixels = [str(i) for i in unique_pixels.tolist()]
    return [file_path, num_unique_pixels, unique_pixels]


def calculate_channel_entropy_vectorized(image: np.ndarray) -> float:
    """
    Vectorized entropy calculation for image channels.

    Computes the average entropy across all channels of an image.

    Args:
        image: Input image as numpy array (grayscale or RGB)

    Returns:
        Average entropy value
    """
    if len(image.shape) == 3:
        # RGB image
        entropies = []
        for channel in range(image.shape[2]):
            hist = np.histogram(image[:, :, channel], bins=256, range=(0, 255))[0]
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropies.append(entropy)
        return np.mean(entropies)
    else:
        # Grayscale image
        hist = np.histogram(image, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-10))


def calculate_rgb_entropy(data):
    """
    Optimized RGB entropy calculation.

    Calculates the entropy of RGB channels for an image.

    Args:
        data: Dictionary containing image data with 'data_samples' key

    Returns:
        List containing [file_path, entropy, [entropy]]
    """
    file_path = data["data_samples"].img_path
    img = cv2.imread(file_path)

    entropy = calculate_channel_entropy_vectorized(img)
    return [file_path, entropy, [entropy]]


def only_cal_diversity_number(
    name: str, save_root: str, data_root: str, data_prefix: Dict, pipeline: List
):
    """
    Optimized diversity number calculation.

    Computes various diversity metrics and caches results for efficiency.

    Args:
        name: Type of diversity to calculate ('class', 'pixel', or 'entropy')
        save_root: Root directory for saving results
        data_root: Root directory of dataset
        data_prefix: Data prefix configuration
        pipeline: Processing pipeline configuration

    Returns:
        Tuple of (mean_diversity, aggregate_metric)
        - For 'class' and 'pixel': (mean, total_unique_count)
        - For 'entropy': (mean, variance)
    """
    from mmseg.datasets import CityscapesDataset, ADE20KDataset

    Path(save_root).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_root) / f"{name}_record.pth"

    try:
        pic_info = torch.load(save_path)
    except Exception as e:
        print(f"{e}, rebuilding file...")

        func_map = {
            "class": compute_pic_class,
            "pixel": compute_pic_pixel,
            "entropy": calculate_rgb_entropy,
        }

        if name not in func_map:
            print(f"Invalid diversity name. Current name is: {name}")
            return None, None

        # Determine dataset type and compute results
        results = []

        if "city" in data_root:
            dataset = CityscapesDataset(
                data_root=data_root,
                data_prefix=data_prefix,
                test_mode=False,
                pipeline=pipeline,
            )
        elif "ADE" in data_root:
            dataset = ADE20KDataset(
                data_root=data_root,
                data_prefix=data_prefix,
                test_mode=False,
                reduce_zero_label=True,
                pipeline=pipeline,
            )

        # Compute using multiprocessing
        num_processes = min(min(20, cpu_count()), len(dataset) // 10 + 1)
        chunk_size = max(1, len(dataset) // (num_processes * 4))

        with Pool(processes=num_processes) as pool:
            results.extend(pool.map(func_map[name], dataset, chunksize=chunk_size))

        torch.save(results, save_path)
        pic_info = results

    pic_info_df = pd.DataFrame(pic_info, columns=["Path", "Results_Number", "results"])

    data = pic_info_df["results"].tolist()
    res = []
    for i in data:
        res.extend(i)

    if name != "entropy":
        res = set(res)
        return pic_info_df["Results_Number"].mean(), len(res)
    else:
        return pic_info_df["Results_Number"].mean(), np.var(res, ddof=0)