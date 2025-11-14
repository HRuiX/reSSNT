from mmengine.registry import init_default_scope
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import DATASETS
import copy
from PIL import Image
import numpy as np
import torch
import cv2
import os
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import threading
from pathlib import Path
from image_transforms import ImageTransforms, comp_plot
import time
import logging
import mmcv
import utility
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

init_default_scope('mmseg')

# Global thread pool to avoid frequent creation/destruction
_write_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="file_writer")
_pending_writes = []  # Track pending write tasks


class FuzzImageData():
    def __init__(self, img_path, seg_map_path, img, idx, image_transforms=None):
        self.img_path = img_path
        self.seg_path = seg_map_path
        self.img = img
        self.mask = cv2.imread(seg_map_path, cv2.IMREAD_GRAYSCALE)
        self.image_transforms = image_transforms if image_transforms is not None else ImageTransforms()
        self.ori_img = img
        self.ori_mask = cv2.imread(seg_map_path, cv2.IMREAD_GRAYSCALE)
        self.trans_cnt = 0
        self.img_idx = idx
        self._path_cache = {}

    def Img_Mutate(self, TRY_NUM, top_save_path, ori_save_path, epoch):
        I_mutated = self.image_transforms(image=self.img, mask=self.mask, TRY_NUM=TRY_NUM)
        if I_mutated is None:
            return None

        self.trans_cnt += 1

        ufzz_img = FuzzImageData(self.img_path, self.seg_path, self.img, self.img_idx, self.image_transforms)
        ufzz_img.img = I_mutated[0]
        ufzz_img.mask = I_mutated[1]
        ufzz_img.image_transforms = self.image_transforms
        ufzz_img.ori_img = self.ori_img
        ufzz_img.ori_mask = self.ori_mask
        ufzz_img.trans_cnt = self.trans_cnt

        return ufzz_img

def process_single_data(args):
    """Function to process single data item for parallel processing"""
    i, dataset = args
    try:

        data = dataset[i]
        seg_map = data['gt_seg_map']
        arr_set = set(seg_map.flatten())

        if len(arr_set) < 3 and 255 in arr_set:
            return None
        return FuzzImageData(data['img_path'], data['seg_map_path'], data['img'], idx=i)

    except Exception as e:
        print(f"Error processing data item {i}: {e}")
        return None


def build_img_np_list_parallel(config_file, random_seed=42, num_workers=None):
    """
    Build image data list in parallel and randomly select specified number of samples
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    if num_workers is None:
        num_workers = min(os.cpu_count(), 20)

    cfg = Config.fromfile(config_file)
    dataloader_cfg = copy.deepcopy(cfg.val_dataloader)

    dataset_cfg = dataloader_cfg.pop('dataset')
    dataset_cfg["pipeline"] = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
    ]

    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()

    dataset_size = len(dataset)
    print(f"Dataset total items: {dataset_size}")

    indices = list(range(dataset_size))

    data_list = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [(i, dataset) for i in indices]
        results = list(executor.map(process_single_data, tasks))
        data_list = [result for result in results if result is not None]

    print(f"Valid data items: {len(data_list)}")

    return data_list

def _read_single_image(img_file, dataset):
    try:
        if dataset == "ade20k":
            mask_file = img_file.replace("leftImg8bit", "annotations").replace(".jpg", ".png")
        else:
            mask_file = img_file.replace("/leftImg8bit", "/gtFine").replace(
                "_leftImg8bit.png", "_gtFine_labelTrainIds.png"
            )

        # 使用mmcv读取图像和mask
        image = mmcv.imread(img_file)
        mask = mmcv.imread(mask_file, flag='grayscale')

        if image is None or mask is None:
            return None

        rel_path = img_file.split("/")[-1]
        return [str(rel_path), image, mask]

    except Exception as e:
        print(f"处理图像 {img_file} 时出错: {e}")
        return None

def get_image_data(dataset, dataset_path_prefix, image_exts=('.jpg', '.jpeg', '.png'),
                   num_workers=None):

    if dataset == "ade20k":
        image_path = f"{dataset_path_prefix}/leftImg8bit/validation"
    else:
        image_path = f"{dataset_path_prefix}/leftImg8bit/val"

    # Collect all image files
    image_files = utility.get_files(image_path)
    image_files = list(set(image_files))  # Deduplicate

    print(f"Found {len(image_files)} image files")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(image_files))

    data_list = []
    failed_count = 0

    # Use process pool for parallel reading
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        read_func = partial(_read_single_image, dataset=dataset)
        futures = {executor.submit(read_func, img_file): img_file
                   for img_file in image_files}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                data_list.append(result)
            else:
                failed_count += 1

    print(f"Dataset total: {len(image_files)}, valid data: {len(data_list)}")
    if failed_count > 0:
        print(f"{failed_count} files failed to read")

    # Sort by filename to ensure consistent order
    data_list.sort(key=lambda x: x[0])

    return data_list



def build_img_np_list(config_file, random_seed=42, use_parallel=True):
    """
    Main function: Select most appropriate implementation based on environment
    """
    return build_img_np_list_parallel(config_file, random_seed)
