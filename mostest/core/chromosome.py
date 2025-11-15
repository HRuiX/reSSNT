"""
Chromosome Encoding and Decoding

New encoding scheme:
Each chromosome is encoded as a one-dimensional vector: [image_index(1), transform1, transform2, ..., transformN]
- First gene: Image index (normalized to 0-1)
- Subsequent genes: Transform sequence
Each transform consists of: [enabled_flag(1), spatial_params(4, optional), transform_params(variable)]

All operations (mutation, crossover) only modify values at corresponding positions
"""
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional, Union
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import uuid
from skimage.metrics import peak_signal_noise_ratio as psnr
from rich.console import Console
from functools import lru_cache
import hashlib

console = Console()

CITYSCAPES_COLORS = np.array([
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],  # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],  # person
    [255, 0, 0],  # rider
    [0, 0, 142],  # car
    [0, 0, 70],  # truck
    [0, 60, 100],  # bus
    [0, 80, 100],  # train
    [0, 0, 230],  # motorcycle
    [119, 11, 32],  # bicycle
], dtype=np.uint8)

try:
    from .transform_registry import (
        TransformConfig, ParamConfig, SpatialMaskedTransform, SpatialParamConfig
    )
except (ImportError, ValueError):
    from transform_registry import (
        TransformConfig, ParamConfig, SpatialMaskedTransform, SpatialParamConfig
    )


class Chromosome:
    """
    Chromosome Class - Encodes image transformation sequence

    New Encoding format:
    One-dimensional vector: [image_idx, T1_enable, T1_bbox(optional), T1_params..., T2_enable, T2_bbox(optional), T2_params..., ...]

    - First gene (index 0): Image index (normalized to 0-1, actual index = int(gene * num_images))
    - Subsequent genes: Transform sequence

    Number of genes per transform:
    - Without spatial localization: 1 (enable) + num_params
    - With spatial localization: 1 (enable) + 4 (bbox: x1,y1,x2,y2) + num_params
    """

    def __init__(
        self,
        transform_configs: List[TransformConfig],
        genes: np.ndarray = None,
        spatial_enabled: bool = False,
        single_transform_init: bool = True,
        num_images: int = 1
    ):
        """
        Initialize chromosome

        Args:
            transform_configs: List of available transform configurations
            genes: Gene sequence, randomly initialized if None
            spatial_enabled: Whether to enable spatial localization (bbox)
            single_transform_init: Whether to enable only one transform during initialization (default True)
            num_images: Total number of available images, used for normalizing image index gene
        """
        self.transform_configs = transform_configs
        self.spatial_enabled = spatial_enabled
        self.num_transforms = len(transform_configs)
        self.num_images = num_images

        # Calculate number of genes needed for each transform
        self.genes_per_transform = []
        for _, config in transform_configs.items():
            gene_count = 1 + config.num_params
            if spatial_enabled: gene_count += 4
            self.genes_per_transform.append(gene_count)

        self.total_genes = 1 + sum(self.genes_per_transform)

        self._gene_index_cache = {}
        cumsum = 1  # Start from 1 (index 0 is image index)
        for i in range(len(self.genes_per_transform)):
            start_idx = cumsum
            end_idx = start_idx + self.genes_per_transform[i]
            self._gene_index_cache[i] = (start_idx, end_idx)
            cumsum = end_idx

        if genes is None:
            self.genes = np.random.uniform(0, 1, self.total_genes)

            if single_transform_init and self.num_transforms > 0:
                # Disable all transforms initially (0.3-0.45)
                for i in range(self.num_transforms):
                    start_idx, _ = self._gene_index_cache[i]
                    self.genes[start_idx] = np.random.uniform(0.3, 0.45)

                # Enable one random transform
                selected_transform = np.random.randint(0, self.num_transforms)
                start_idx, _ = self._gene_index_cache[selected_transform]
                self.genes[start_idx] = np.random.uniform(0.7, 1.0)
        else:
            assert len(genes) == self.total_genes, \
                f"Gene length mismatch: expected {self.total_genes}, got {len(genes)}"
            self.genes = genes.copy()

        self.objectives: List[float] = [0.0, 0.0, 0.0]
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.reference_point_distance: float = float('inf')

        self.ori_file_name: str = None
        self.muta_data: Dict = None

        self.params: Dict = None
        self.enabled_transforms: List[str] = []

        # self._cached_transform = None
        # self._cached_genes_hash = None

    @staticmethod
    def get_gene_index_static(transform_idx: int, genes_per_transform: List[int]) -> Tuple[int, int]:
        """
        Static method: Get start and end index of specified transform in gene vector
        Used when self is not available during initialization

        Note: Returned indices already account for image index gene (index 0),
        so transform genes start from index 1

        Args:
            transform_idx: Transform index
            genes_per_transform: List of gene counts for each transform

        Returns:
            (start_idx, end_idx): Index range in gene vector (already offset +1)
        """
        # Offset +1 because first gene (index 0) is image index
        start_idx = 1 + sum(genes_per_transform[:transform_idx])
        end_idx = start_idx + genes_per_transform[transform_idx]
        return start_idx, end_idx

    def get_gene_index(self, transform_idx: int) -> Tuple[int, int]:
        """
        Get start and end index of specified transform in gene vector

        Note: Returned indices already account for image index gene (index 0),
        so transform genes start from index 1

        Args:
            transform_idx: Transform index [0, num_transforms-1]

        Returns:
            (start_idx, end_idx): Index range in gene vector (already offset +1)
        """
        return self._gene_index_cache[transform_idx]

    def get_image_index(self) -> int:
        """
        Get chromosome's image index

        Returns:
            Image index [0, num_images-1]
        """
        normalized_index = np.clip(self.genes[0], 0.0, 1.0)
        actual_index = min(int(normalized_index * self.num_images), self.num_images - 1)
        actual_index = max(0, actual_index)
        return actual_index

    def set_image_index(self, image_idx: int):
        """
        Set chromosome's image index

        Args:
            image_idx: Image index [0, num_images-1]
        """
        assert 0 <= image_idx < self.num_images, \
            f"Image index out of range: {image_idx} (range: [0, {self.num_images-1}])"
        self.genes[0] = (image_idx + 0.5) / self.num_images

    def _compute_genes_hash(self) -> str:
        """
        OPTIMIZATION: Use fast hash for cache invalidation
        Faster than comparing entire arrays
        """
        # Use tobytes() for consistent hashing
        return hashlib.md5(self.genes.tobytes()).hexdigest()

    def decode(self) -> A.Compose:
        """
        Decode chromosome to Albumentations transform sequence

        Cache optimization: If genes haven't changed, return cached transform directly

        Returns:
            A.Compose: Composed transform object
        """
        # current_hash = self._compute_genes_hash()
        # if self._cached_genes_hash == current_hash and self._cached_transform is not None:
        #     return self._cached_transform

        transforms_list = []
        gene_idx = 1

        for i, config in self.transform_configs.items():
            enable_flag = self.genes[gene_idx]
            gene_idx += 1

            if enable_flag < 0.5:
                gene_idx += self.genes_per_transform[i] - 1
                continue

            bbox_genes = None
            if self.spatial_enabled:
                bbox_genes = self.genes[gene_idx: gene_idx + 4]
                gene_idx += 4

            param_genes = self.genes[gene_idx: gene_idx + config.num_params]
            gene_idx += config.num_params
            params_dict = config.decode_params(param_genes)

            try:
                transform = config.create_transform(params_dict)

                if self.spatial_enabled and bbox_genes is not None:
                    x1, y1, x2, y2 = bbox_genes
                    if x1 > x2:
                        x1, x2 = x2, x1
                    if y1 > y2:
                        y1, y2 = y2, y1

                    transform = SpatialMaskedTransform(
                        transform,
                        bbox_normalized=(float(x1), float(y1), float(x2), float(y2))
                    )

                transforms_list.append(transform)

            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] Failed to create {config.name}: [dim]{e}[/dim]")
                continue

        composed_transform = A.Compose(transforms_list)
        return composed_transform

    def get_transform_summary(self) -> List[Dict]:
        """
        Get transform summary (for debugging and visualization)

        Returns:
            Detailed information for each enabled transform
        """
        summary = []
        gene_idx = 1

        for i, config in self.transform_configs.items():
            enable_flag = self.genes[gene_idx]
            gene_idx += 1

            if enable_flag < 0.5:
                gene_idx += self.genes_per_transform[i] - 1
                continue

            info = {
                "name": config.name,
                "enable_flag": float(enable_flag)
            }

            if self.spatial_enabled:
                bbox_genes = self.genes[gene_idx: gene_idx + 4]
                gene_idx += 4
                x1, y1, x2, y2 = bbox_genes
                info["bbox"] = (float(x1), float(y1), float(x2), float(y2))

            param_genes = self.genes[gene_idx: gene_idx + config.num_params]
            gene_idx += config.num_params

            params_dict = config.decode_params(param_genes)
            info["params"] = params_dict

            summary.append(info)

        return summary

    def update_params_info(self):
        """
        Update params and enabled_transforms attributes
        Extract and populate this information from genes, used for saving and logging
        """
        self.enabled_transforms = []
        self.params = {}

        gene_idx = 1

        for i, config in self.transform_configs.items():
            enable_flag = self.genes[gene_idx]
            gene_idx += 1

            if enable_flag < 0.5:
                gene_idx += self.genes_per_transform[i] - 1
                continue

            self.enabled_transforms.append(config.name)

            if self.spatial_enabled:
                bbox_genes = self.genes[gene_idx: gene_idx + 4]
                gene_idx += 4
                x1, y1, x2, y2 = bbox_genes
                self.params[f"{config.name}_bbox"] = {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2)
                }

            param_genes = self.genes[gene_idx: gene_idx + config.num_params]
            gene_idx += config.num_params

            params_dict = config.decode_params(param_genes)
            self.params[config.name] = params_dict

    @staticmethod
    def colorize_seg_map(seg_map, colors=CITYSCAPES_COLORS, ignore_index=255):
        """Colorize segmentation map"""
        h, w = seg_map.shape
        colored_seg = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in range(len(colors)):
            mask = seg_map == class_id
            colored_seg[mask] = colors[class_id]

        ignore_mask = seg_map == ignore_index
        colored_seg[ignore_mask] = [0, 0, 0]

        return colored_seg

    def apply_transform(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.mask_ = """
        Apply transform to image and mask

        Args:
            image: Input image (H, W, 3)
            mask: Input mask (H, W)

        Returns:
            transformed_image, transformed_mask
        """
        transform = self.decode()

        try:
            transformed = transform(image=image, mask=mask)
            return transform, transformed['image'], transformed['mask']
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Transform failed: [dim]{e}[/dim]")
            console.print(f"  [dim]Transform: {transform}[/dim]")
            # Return copy to avoid unintended modifications
            return None, image.copy(), mask.copy()

    def enforce_max_transforms(self, max_transforms: int = 3, enable_threshold: float = 0.5):
        """
        Enforce maximum number of enabled transforms
        If multiple transforms are enabled and exceed max_transforms, randomly select max_transforms to keep, disable others
        If no transforms are enabled, randomly enable one

        Args:
            max_transforms: Maximum number of allowed enabled transforms (default 3)
            enable_threshold: Threshold for transform enablement (default 0.5)
        """
        # Find all enabled transform indices
        enabled_indices = []
        for i in range(self.num_transforms):
            start_idx, _ = self.get_gene_index(i)
            if self.genes[start_idx] >= enable_threshold:
                enabled_indices.append(i)

        if len(enabled_indices) == 0:
            # No enabled transforms, randomly enable one
            selected = np.random.randint(0, self.num_transforms)
            start_idx, _ = self.get_gene_index(selected)
            self.genes[start_idx] = np.random.uniform(0.65, 1.0)
        elif len(enabled_indices) > max_transforms:
            # Too many enabled, randomly keep max_transforms
            selected_indices = np.random.choice(enabled_indices, size=max_transforms, replace=False)
            for i in enabled_indices:
                start_idx, _ = self.get_gene_index(i)
                if i in selected_indices:
                    if self.genes[start_idx] < enable_threshold:
                        self.genes[start_idx] = np.random.uniform(0.65, 1.0)
                else:
                    self.genes[start_idx] = np.random.uniform(0.0, 0.45)


    def copy(self) -> 'Chromosome':
        """Create chromosome copy"""
        new_chrom = Chromosome(
            self.transform_configs,
            genes=self.genes.copy(), 
            spatial_enabled=self.spatial_enabled,
            num_images=self.num_images
        )
        new_chrom.objectives = self.objectives.copy()
        new_chrom.rank = self.rank
        new_chrom.crowding_distance = self.crowding_distance
        new_chrom.reference_point_distance = self.reference_point_distance
        new_chrom.muta_data = self.muta_data
        new_chrom.ori_file_name = self.ori_file_name
        new_chrom.params = self.params.copy() if self.params else None
        new_chrom.enabled_transforms = self.enabled_transforms.copy()
        return new_chrom

    def __repr__(self) -> str:
        enabled_count = sum(1 for i in range(self.num_transforms)
                          if self.genes[self.get_gene_index(i)[0]] >= 0.5)
        return (f"Chromosome(transforms={enabled_count}/{self.num_transforms}, "
                f"objectives={self.objectives}, rank={self.rank})")

    def check_chromosome_validity(self, datalist, psnr_threshold) -> bool:
        """
        检查种群中所有染色体的有效性

        Args:
            population: 种群

        Returns:
            是否所有染色体均有效
        """
        seed_idx = self.get_image_index()
        file_name, seed_image, seed_mask = datalist[seed_idx]
        transform, mutated_image, mutated_mask = self.apply_transform(seed_image, seed_mask)
        if (transform is None) or np.all(mutated_image == seed_image) or (
                psnr(seed_image, mutated_image) < psnr_threshold):
            return False

        finames = file_name.split("_")
        uuids = str(uuid.uuid4())[:8]
        finames = f"{finames[0]}_{str(transform[0]).split('(')[0]}_{uuids}_{'_'.join(finames[1:])}"
        muta_data = {
            "seed_idx": seed_idx,
            "uuid": uuids,
            "muta_image": mutated_image,
            "muta_mask": mutated_mask,
            "file_name": finames,
        }

        self.ori_file_name = file_name
        self.muta_data = muta_data
        return True

    @staticmethod
    def create_random_population(
        population_size: int,
        transform_configs: List[TransformConfig],
        datalist,
        psnr_threshold,
        spatial_enabled: bool = False,
        single_transform_init: bool = True,
        num_images: int = 1
    ) -> List['Chromosome']:
        """
        Create random initial population

        Args:
            population_size: Population size
            transform_configs: Transform configuration list
            spatial_enabled: Whether to enable spatial localization
            single_transform_init: Whether to enable only one transform during initialization (default True)
            num_images: Total number of available images

        Returns:
            List of Chromosome objects
        """
        population = []
        pbar = tqdm(total=population_size,desc="Buliding population")
        while len(population) < population_size:
            chromosome = Chromosome(transform_configs,spatial_enabled=spatial_enabled,single_transform_init=single_transform_init,num_images=num_images)
            valid = Chromosome.check_chromosome_validity(chromosome,datalist,psnr_threshold)
            if valid:
                population.append(chromosome)
                pbar.update()
        return population[:population_size]

from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]