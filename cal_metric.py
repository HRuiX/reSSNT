import os
import torch
from mmseg.datasets import CityscapesDataset, ADE20KDataset
from tqdm.auto import tqdm
from mmseg.apis import inference_model, init_model
import numpy as np
from mmengine.config import Config
import gc
import utility
import torch
from mmengine.runner import Runner
from mmengine.config import Config
import os
import os.path as osp
from hook import OutputSizeHook, SaveLayerInputOutputHook
import utility
from functools import partial
import cal_cov
from torch.utils.data import DataLoader
import analyse
import perpa_data
from rich import print
import cal_diversity
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import models
import torch.nn as nn


def _build_dataset_for_coverage(data_root, dataset):
        """缓存的数据集构建"""
        if dataset == "cityscapes":
            # data_root = f"{data_root}/cityscapes"
            data_prefix = dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val')
            pipeline = [
                dict(type='LoadImageFromFile'),
                dict(keep_ratio=True, scale=(2048, 1024), type='Resize'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs')
            ]
            return CityscapesDataset(
                data_root=data_root, 
                data_prefix=data_prefix, 
                test_mode=False, 
                pipeline=pipeline
            )
        else:
            # data_root = f"{data_root}/ADEChallengeData2016"
            data_prefix = dict(img_path='leftImg8bit/validation/', seg_map_path='annotations/validation/')
            pipeline = [
                dict(type='LoadImageFromFile'),
                dict(keep_ratio=True, scale=(2048, 512), type='Resize'),
                dict(reduce_zero_label=True, type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ]
            return ADE20KDataset(
                data_root=data_root, 
                data_prefix=data_prefix, 
                test_mode=False, 
                reduce_zero_label=True,
                pipeline=pipeline
            )



def _extract_name_from_key(dataset, key):
    """提取文件名"""
    if dataset == "ade20k":
        return key.split("_")[2].split(".")[0]
    elif dataset == "cityscapes":
        return key.split(".")[0]
    return key
    
DATASETS = [[], ["ade20k"], ["cityscapes"], ["ade20k", "cityscapes"]]
MODEL_TYPE = ["", "CNN", "Transformer", "Other"]

datasets_idx = 3
for model_type_idx in [1, 2, 3]:
    model_type = MODEL_TYPE[model_type_idx]
    for dataset_name in DATASETS[datasets_idx]:
        model_infos, dataset_path_prefix = utility.get_file_device(dataset_name, model_type)
        for model_info in model_infos:
            batch_size = 1
            model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
            
            runner = ""

            """计算原始指标（优化版本）"""
            dataset = _build_dataset_for_coverage(dataset_path_prefix, dataset_name)
            cfg = Config.fromfile(config)

            dataloader = cfg.val_dataloader
            dataloader.dataset = dataset
            dataloader.batch_size = batch_size
            
            cfg = Config.fromfile(config)

            cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
            cfg.load_from = checkpoint
            cfg.launcher = "none"

            runner = Runner.from_cfg(cfg)
            runner.model.cfg = cfg

            if hasattr(dataloader, 'batch_sampler'):
                dataloader.batch_sampler.batch_size = batch_size

            dataloader = runner.build_dataloader(dataloader)
            evaluator = runner.build_evaluator(cfg.val_evaluator)

            if hasattr(dataloader.dataset, 'metainfo'):
                evaluator.dataset_meta = dataloader.dataset.metainfo

            runner.model.eval()
            ori_metrics = {}

            with torch.no_grad():
                for idx, data_batch in enumerate(tqdm(dataloader, desc="计算原始指标",leave=False)):
                    # 内存管理
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    outputs = runner.model.val_step(data_batch)
                    evaluator.process(data_samples=outputs, data_batch=data_batch)
                    metrics = evaluator.evaluate(len(data_batch))
                    
                    key = data_batch['data_samples'][0].img_path.split("/")[-1]
                    name = _extract_name_from_key(dataset, key)
                    ori_metrics[name] = metrics

                    # 清理中间变量
                    del outputs, data_batch, metrics
            prefix = f"./temp/metric/{model_name}/"
            utility.build_path(prefix)
            
            torch.save(ori_metrics,f"{prefix}/{dataset_name}-ori_metrics.pth")

