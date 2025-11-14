import os
import torch
from typing import Optional, Sequence, Dict, Any
from mmengine.hooks.hook import DATA_BATCH
import torch.nn as nn
from mmengine.hooks import Hook
from mmseg.datasets import CityscapesDataset, ADE20KDataset
from tqdm.auto import tqdm
from mmseg.apis import inference_model, init_model
import numpy as np
from rich.console import Console
from mmengine.config import Config
import gc
import multiprocessing as mp
from copy import deepcopy
from functools import partial
from functools import lru_cache
import weakref
import copy
import tempfile
import shutil
from collections import deque
import psutil
import threading
import time
import utility
console = Console()
mp.set_start_method('spawn', force=True)


class OutputSizeHook(Hook):
    def __init__(self, args, demo_path=None):
        super().__init__()
        self.args = args
        self.hooks = []
        self.handles = []
        self.output_sizes = {}

        self.layer_dict = args.layer_dict
        self.layer_dict_keys = list(self.layer_dict.keys())
        self.layer_dict_keys_cnt = 0
        self.run_for_size(demo_path=demo_path)

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            for name, m in self.args.model.named_modules():
                if m is module:
                    name = f"{self.layer_dict_keys_cnt+1}-{name}"
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d) or isinstance(m,
                                                                                                                      nn.GroupNorm):
                        out_size = output.size()[1]
                    elif isinstance(m, nn.Linear):
                        out_size = m.out_features
                    elif isinstance(m, nn.LayerNorm):
                        out_size = m.normalized_shape[0]
                    else:
                        print(m)
                    # print(name,module,output.size(),out_size)
                    self.output_sizes[name] = {"Output": [out_size], "Input": input[0].size()}
                    self.layer_dict_keys_cnt += 1
                    break

        return hook

    def check_demo_path(self, demo_path):
        if demo_path is not None:
            return {'img_path': demo_path}
        elif self.args.dataset == "ade20k":
            return {'img_path': "./demo/demo-ade20k.jpg"}
        elif self.args.dataset == "cityscapes":
            return {'img_path': "./demo/demo.png"}
        elif self.args.task_type == "DNN":
            input_size = (1, 3, 32, 32)
            random_data = torch.randn(input_size).to(self.device)

    def run_for_size(self, demo_path=None):
        path = self.args.TOOL_LOG_FILE_PATH + "/output_sizes.pth"

        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):  # 只对 nn.Module 注册 hook
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

        data = self.check_demo_path(demo_path)

        self.layer_dict_keys_cnt = 0

        try:
            processed_data = self.args.transforms(data)
            input_tensor = processed_data['inputs']
            img_np = input_tensor.numpy()
            img = np.transpose(img_np, (1, 2, 0))

            inference_model(self.args.model, img)
        except:
            self.args.model(data['img_path'])

        torch.save(self.output_sizes, path)
        self.layer_dict_keys = list(self.output_sizes.keys())

        # 在运行结束后移除所有 hook
        for handle in self.handles:
            handle.remove()

        # 自定义 Hook 用于保存每一层的输入和输出


class SaveLayerInputOutputHook(Hook):
    def __init__(self, args, task_type="SSN", **kwargs):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.datasets = args.dataset
        self.TOOL_LOG_FILE_PATH = args.TOOL_LOG_FILE_PATH
        self.coverages = args.coverages
        self.task_type = task_type

        self.output_sizes = args.output_sizes
        self.layer_dict = args.layer_dict
        self.layer_dict_keys = list(args.output_sizes.keys())
        torch.save(self.layer_dict_keys, f"{self.TOOL_LOG_FILE_PATH}/layer_dict_keys.pth")
        self.layer_dict_keys_cnt = 0

        self.hooks = []
        self.handles = []
        self.layer_output_input_dict = {}
        self.output = None
        self.type = args.mode

        if task_type == "SSN":
            self.file_name = args.config.split("/")[-1].replace(".py", "")

    def before_run(self, runner):
        if self.type == "train":
            path = f"{self.TOOL_LOG_FILE_PATH}/coverages/"

        # 为每一层注册 forward hook
        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):  # 只对 nn.Module 注册 hook
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def before_val(self, data) -> None:
        if len(self.coverages) > 2:
            N = len(self.coverages)
        else:
            N = self.args.coverages_setting

        # if self.type != "choice_test":
        if self.task_type == "SSN":
            print(f"The {self.file_name} has {len(data.val_dataloader)} in {self.args.coverages_setting} coverages.")
            self.progress = tqdm(total=len(data.val_dataloader),
                                    desc=f"   → Analysis {self.file_name} data in {N} coverages with {self.type}",
                                    leave=False)
        else:
            self.progress = tqdm(total=len(data),
                                    desc=f"   → Analysis data in {N} coverages with {self.type}")

    def before_val_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None) -> None:
        self.input_cnt = batch_idx
        self.layer_dict_keys_cnt = 0
        if self.task_type == "SSN":
            if len(data_batch["data_samples"]) == 1:
                self.input_name = (data_batch["data_samples"][0].img_path, data_batch['data_samples'][0].seg_map_path)
            else:
                self.input_name = batch_idx
        else:
            self.input_name = batch_idx

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            assert len(input[0]) == output.shape[0], f"input_names is {self.input_name}, but input has {len(input[0])}, output has {output.shape}."
            
            if output.size()[0] != 1:
                tmp_output = output.unsqueeze(0)
                tmp_output = tmp_output.mean((1, 2))
            else:
                tmp_output = copy.deepcopy(output)

                if len(tmp_output.size()) == 4 and not (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((2, 3))
                elif len(tmp_output.size()) == 4 and (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((1, 2))
                elif len(tmp_output.size()) == 3:  # (H, W, C)
                    tmp_output = tmp_output.mean(dim=1)
                elif len(tmp_output.size()) == 2 and (isinstance(module, nn.Linear)) and self.task_type == "DNN":  # (N, K)
                    None
                else:
                    print(module)
                    print(tmp_output.size())
                    print(len(tmp_output.size()))
                    print(isinstance(module, nn.Linear))
                    print(self.task_type)
                    exit()

            # check name
            for name, m in self.args.model.named_modules():
                if m is module:
                    name = f"{self.layer_dict_keys_cnt + 1}-{name}"
                    try:
                        if name == self.layer_dict_keys[self.layer_dict_keys_cnt]:
                            break
                    except Exception as e:
                        print(
                            f"[Error] The pic name is {self.input_name}. {name} Not in layer_dict_keys. The exception is: {e}. We will drop it.")

                if self.input_name in self.layer_output_input_dict.keys():
                    try:
                        self.layer_output_input_dict[self.input_name][
                            self.layer_dict_keys[self.layer_dict_keys_cnt]] = tmp_output.detach()
                    except Exception as e:
                        print(e)
                else:
                    self.layer_output_input_dict[self.input_name] = {
                        self.layer_dict_keys[self.layer_dict_keys_cnt]: tmp_output.detach()}

            if self.output == None or not torch.equal(self.output, tmp_output):
                self.layer_dict_keys_cnt += 1

            self.output = tmp_output

        return hook

    def after_val_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        self.layer_dict_keys_cnt = 0

        try:
            if self.task_type == "SSN":
                self.progress.update(len(data_batch["inputs"]))
            else:
                self.progress.update()
        except Exception as e:
            print(e)

        for key, covers in self.coverages.items():
            for cover in covers:
                if self.type == "train":
                    cover.build(self.layer_output_input_dict[self.input_name])
                elif "test" in self.type:
                    cover.step(self.layer_output_input_dict[self.input_name], self.input_name)

        self.layer_output_input_dict = {}

    def after_val(self, runner) -> None:
        for key, covers in self.coverages.items():
            for cover in covers:
                if self.type == "train":
                    try:
                        cover.save_build()
                    except:
                        if key.split("-")[0] not in ["NC", "NLC", "TKNC", "TKNP", "CC"]:
                            print(f"Failed to save build related files. \n"
                                  f"{key} may not require a build file, or for other reasons, please verify.\n")
                elif "test" in self.type:
                    cover.save(self.type)

    def after_run(self, model):
        self.layer_output_input_dict = {}
        self.layer_dict_keys_cnt = 0
        self.output = None

        for key, covers in self.coverages.items():
            for cover in covers:
                print(f"Coverage {key} is ", cover.current)

        for handle in self.handles:
            handle.remove()


class CoverageGainHook(Hook):
    """优化的覆盖率增益Hook"""

    def __init__(self, ori_save_path, muta_save_path, args):
        super().__init__()
        self.args = args
        # 性能优化
        self.batch_size = 1  # 小批量处理以节省内存
        self.memory_cleanup_interval = 30
        self.TOOL_LOG_FILE_PATH = args.TOOL_LOG_FILE_PATH
        self.coverages = args.coverages
        self.layer_dict = args.layer_dict
        self.layer_dict_keys = list(args.output_sizes.keys())
        self.file_name = args.config.split("/")[-1].replace(".py", "")
        self.type = args.mode
        self.output = None

        # 数据加载器和评估器
        self.dataloader, self.evaluator = self._build_data_evaluator(muta_save_path)
        self.ori_metrics = self._calculate_original_metrics(ori_save_path)

        # Hook管理
        self.handles = []
        self.batch_cache = {}

        # ✅ 保持原始属性兼容性
        self.layer_output_input_dict = {}
        self.layer_dict_keys_cnt = 0

    @lru_cache(maxsize=2)
    def _build_dataset_for_coverage(self, data_root):
        """缓存的数据集构建"""
        if self.args.dataset == "cityscapes":
            data_root = f"{data_root}/cityscapes"
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
            data_root = f"{data_root}/ADEChallengeData2016"
            data_prefix = dict(img_path='images/validation/', seg_map_path='annotations/validation/')
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

    def _build_data_evaluator(self, data_root):
        """构建数据加载器和评估器"""
        dataset = self._build_dataset_for_coverage(data_root)
        cfg = Config.fromfile(self.args.config)

        dataloader = cfg.val_dataloader
        dataloader.dataset = dataset
        dataloader.batch_size = self.batch_size

        if hasattr(dataloader, 'batch_sampler'):
            dataloader.batch_sampler.batch_size = self.batch_size

        dataloader = self.args.runner.build_dataloader(dataloader)
        evaluator = self.args.runner.build_evaluator(cfg.val_evaluator)

        if hasattr(dataloader.dataset, 'metainfo'):
            evaluator.dataset_meta = dataloader.dataset.metainfo

        return dataloader, evaluator
    
    

    def _calculate_original_metrics(self, data_root):
        """计算原始指标（优化版本）"""
        try:
            ori_metrics = torch.load(f"./temp/metric/{self.args.model_name}/{self.args.model_name}-ori_metrics.pth")
        except:
            dataset = self._build_dataset_for_coverage(data_root)
            cfg = Config.fromfile(self.args.config)

            dataloader = cfg.val_dataloader
            dataloader.dataset = dataset
            dataloader.batch_size = self.batch_size

            if hasattr(dataloader, 'batch_sampler'):
                dataloader.batch_sampler.batch_size = self.batch_size

            dataloader = self.args.runner.build_dataloader(dataloader)
            evaluator = self.args.runner.build_evaluator(cfg.val_evaluator)

            if hasattr(dataloader.dataset, 'metainfo'):
                evaluator.dataset_meta = dataloader.dataset.metainfo

            self.args.model.eval()
            ori_metrics = {}

            with torch.no_grad():
                for idx, data_batch in enumerate(tqdm(dataloader, desc="计算原始指标", leave=False)):
                    # 内存管理
                    if idx % self.memory_cleanup_interval == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    outputs = self.args.model.val_step(data_batch)
                    evaluator.process(data_samples=outputs, data_batch=data_batch)
                    metrics = evaluator.evaluate(len(data_batch))

                    key = data_batch['data_samples'][0].img_path.split("/")[-1]
                    name = self._extract_name_from_key(key)
                    ori_metrics[name] = metrics

                    # 清理中间变量
                    del outputs, data_batch, metrics
                    
        from pathlib import Path    
        os.makedirs(f"./temp/metric/{self.args.model_name}", exist_ok=True)    
        torch.save(ori_metrics,f"./temp/metric/{self.args.model_name}/{self.args.model_name}-ori_metrics.pth")
        
        return ori_metrics

    def _extract_name_from_key(self, key):
        """提取文件名"""
        if self.args.dataset == "ade20k":
            return key.split("_")[2].split(".")[0]
        elif self.args.dataset == "cityscapes":
            return key.split(".")[0]
        return key

    def _build_hooks(self):
        """构建hooks"""
        # 为每一层注册 forward hook
        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):  # 只对 nn.Module 注册 hook
                handle = module.register_forward_hook(self._create_hook_fn(name))
                self.handles.append(handle)

    def _create_hook_fn(self, module_id):
        """创建优化的hook函数"""

        def hook(module, input, output):
            # reshape output
            if output.size()[0] != 1:
                tmp_output = output.unsqueeze(0)
                tmp_output = tmp_output.mean((1, 2))
            else:
                tmp_output = output.clone().detach()
                if len(tmp_output.size()) == 4 and not (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((2, 3))
                elif len(tmp_output.size()) == 4 and (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((1, 2))
                elif len(tmp_output.size()) == 3:  # (H, W, C)
                    tmp_output = tmp_output.mean(dim=1)
                else:
                    print(module)

            # check name
            for name, m in self.args.model.named_modules():
                if m is module:
                    name = f"{self.layer_dict_keys_cnt + 1}-{name}"
                    try:
                        if name == self.layer_dict_keys[self.layer_dict_keys_cnt]:
                            break
                    except Exception as e:
                        print(
                            f"[Error] The pic name is {self.input_name}. {name} Not in layer_dict_keys. The exception is: {e}. We will drop it.")

            if self.input_name in self.layer_output_input_dict.keys():
                try:
                    self.layer_output_input_dict[self.input_name][
                        self.layer_dict_keys[self.layer_dict_keys_cnt]] = tmp_output
                except Exception as e:
                    print(e)

            else:
                self.layer_output_input_dict[self.input_name] = {
                    self.layer_dict_keys[self.layer_dict_keys_cnt]: tmp_output}

            if self.output == None or not torch.equal(self.output, tmp_output):
                self.layer_dict_keys_cnt += 1

            self.output = tmp_output

        return hook

    def __call__(self):
        """主执行函数"""
        self._build_hooks()

        cov_use_for_data_path = []
        ad_cov_use_for_data_path = []
        self.args.model.eval()

        progress = tqdm(total=len(self.dataloader), desc="处理变异图片", leave=False)

        with torch.no_grad():
            for idx, data_batch in enumerate(self.dataloader):

                self.input_name = data_batch["data_samples"][0].img_path.split("/")[-1][:-4]

                # 模型推理
                outputs = self.args.model.val_step(data_batch)
                self.evaluator.process(data_samples=outputs, data_batch=data_batch)
                metrics = self.evaluator.evaluate(len(data_batch))

                # 获取路径信息
                path = data_batch['data_samples'][0].img_path
                seg_path = data_batch['data_samples'][0].seg_map_path
                key_path = data_batch['data_samples'][0].img_path.split("/")[-1]

                # 处理覆盖率
                for key, covers in self.coverages.items():
                    try:
                        cover = covers[0]
                        _, cove_dict = cover.calculate(self.layer_output_input_dict[self.input_name])
                        gain = cover.gain(cove_dict)
                        
                        if self._has_coverage_gain(gain):
                            cover.update(cove_dict, gain)
                            cov_use_for_data_path.append((key, path, seg_path))

                        if self._check_performance_degradation(key_path, metrics):
                            ad_cov_use_for_data_path.append((key, path, seg_path))

                    except Exception as e:
                        console.print(f"[red]Error processing {key}: {e}[/red]")


                # ✅ 清理原始属性以保持兼容性
                self.layer_output_input_dict = {}
                self.layer_dict_keys_cnt = 0
                del outputs, data_batch, metrics

                progress.update()

        self._cleanup_hooks()
        return cov_use_for_data_path, ad_cov_use_for_data_path

    def _has_coverage_gain(self, gain, min_threshold=1e-6):
        if gain is None:
            return False

        if isinstance(gain, tuple):
            return gain[0] > min_threshold
        else:
            return gain > min_threshold

    def _check_performance_degradation(self, key_path, metrics):
        name = self._extract_name_from_path(key_path)
        if name not in self.ori_metrics:
            print(key_path)
            print(name)
            print(f"[red]Original metric for {name} not found. Please check your dataset configuration.[/red]")
            return False

        ori_metric = self.ori_metrics[name]
        return (metrics["aAcc"] < ori_metric["aAcc"] or
                metrics["mIoU"] < ori_metric["mIoU"] or
                metrics["mAcc"] < ori_metric["mAcc"])

    
    def _extract_name_from_path(self, key_path):
        """从路径提取名称"""
        parts = key_path.split(".")[0].split('_')
        position = parts.index('epoch')
        parts = parts[:position] + parts[position + 4:]
        if self.args.dataset == "ade20k":
            return  parts[-1]
        elif self.args.dataset == "cityscapes":
            return "_".join(parts)
        return key_path

    def _emergency_cleanup(self):
        """紧急内存清理"""
        self.batch_cache.clear()

        # ✅ 清理原始属性以保持兼容性
        self.layer_output_input_dict = {}
        self.layer_dict_keys_cnt = 0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _cleanup_hooks(self):
        """清理hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._emergency_cleanup()




# class CoverageAddHook(Hook):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         # 性能优化
#         self.batch_size = 1  # 小批量处理以节省内存
#         self.TOOL_LOG_FILE_PATH = args.TOOL_LOG_FILE_PATH
#         self.coverages = args.coverages
#         self.layer_dict = args.layer_dict
#         self.layer_dict_keys = list(args.output_sizes.keys())
#         self.file_name = args.config.split("/")[-1].replace(".py", "")
#         self.type = args.mode
#         self.output = None

#         # Hook管理
#         self.handles = []
#         self.batch_cache = {}

#         # ✅ 保持原始属性兼容性
#         self.layer_output_input_dict = {}
#         self.layer_dict_keys_cnt = 0
#         self.memory_cleanup_interval = 30
        
   
#     def _build_data_evaluator(self):
#         """构建数据加载器和评估器"""
#         cfg = Config.fromfile(self.args.config)

#         dataloader = cfg.val_dataloader
#         dataloader.batch_size = self.batch_size
#         dataloader.num_workers = 25

#         if hasattr(dataloader, 'batch_sampler'):
#             dataloader.batch_sampler.batch_size = self.batch_size

#         dataloader = self.args.runner.build_dataloader(dataloader)

#         return dataloader


#     def _build_hooks(self):
#         """构建hooks"""
#         # 为每一层注册 forward hook
#         for name, module in self.layer_dict.items():
#             if isinstance(module, nn.Module):  # 只对 nn.Module 注册 hook
#                 handle = module.register_forward_hook(self._create_hook_fn(name))
#                 self.handles.append(handle)

#     def _create_hook_fn(self, module_id):
#         """创建优化的hook函数"""
#         def hook(module, input, output):
#             # reshape output
#             if output.size()[0] != 1:
#                 tmp_output = output.unsqueeze(0)
#                 tmp_output = tmp_output.mean((1, 2))
#             else:
#                 tmp_output = output.clone().detach()
#                 if len(tmp_output.size()) == 4 and not (isinstance(module, nn.Linear)):  # (N, K, H, w)
#                     tmp_output = tmp_output.mean((2, 3))
#                 elif len(tmp_output.size()) == 4 and (isinstance(module, nn.Linear)):  # (N, K, H, w)
#                     tmp_output = tmp_output.mean((1, 2))
#                 elif len(tmp_output.size()) == 3:  # (H, W, C)
#                     tmp_output = tmp_output.mean(dim=1)
#                 else:
#                     print(module)

#             # check name
#             for name, m in self.args.model.named_modules():
#                 if m is module:
#                     name = f"{self.layer_dict_keys_cnt + 1}-{name}"
#                     try:
#                         if name == self.layer_dict_keys[self.layer_dict_keys_cnt]:
#                             break
#                     except Exception as e:
#                         print(
#                             f"[Error] The pic name is {self.input_name}. {name} Not in layer_dict_keys. The exception is: {e}. We will drop it.")

#             if self.input_name in self.layer_output_input_dict.keys():
#                 try:
#                     self.layer_output_input_dict[self.input_name][
#                         self.layer_dict_keys[self.layer_dict_keys_cnt]] = tmp_output
#                 except Exception as e:
#                     print(e)

#             else:
#                 self.layer_output_input_dict[self.input_name] = {
#                     self.layer_dict_keys[self.layer_dict_keys_cnt]: tmp_output}

#             if self.output == None or not torch.equal(self.output, tmp_output):
#                 self.layer_dict_keys_cnt += 1

#             self.output = tmp_output

#         return hook

#     def _build_data_reocrd(self, model, save_path):
#         data_record = {}
#         progress = tqdm(total = len(self.dataloader),desc="Find max gain:")
#         with torch.no_grad():
#             for idx, data_batch in enumerate(self.dataloader):
#                 if idx % self.memory_cleanup_interval == 0:
#                     self._emergency_cleanup()

#                 self.input_name = data_batch["data_samples"][0].img_path

#                 # 模型推理
#                 outputs = model.val_step(data_batch)

#                 # 获取路径信息
#                 path = self.input_name
#                 seg_path = data_batch['data_samples'][0].seg_map_path
                
#                 if f"{self.input_name}" not in data_record:
#                     data_record[self.input_name] = [[self.layer_output_input_dict[self.input_name],path,seg_path]]
#                 else:
#                     data_record[self.input_name].append([self.layer_output_input_dict[self.input_name],path,seg_path])

#                 # ✅ 清理原始属性以保持兼容性
#                 self.layer_output_input_dict = {}
#                 self.layer_dict_keys_cnt = 0
#                 del outputs, data_batch
#                 progress.update()
#         torch.save(data_record,save_path)
#         del data_record
#         gc.collect()



#     def __call__(self):
#         """主执行函数"""
#         self._build_hooks()
#         self.args.model.eval()
#         self.dataloader = self._build_data_evaluator()
#         data_record = {}
#         path = f"{self.args.TOOL_LOG_FILE_PATH}/cov_record/"
#         utility.build_path(path)
#         save_path = f"{path}/{self.args.model_name}_coverage_info.pt"
        
#         if not os.path.exists(save_path):
#             print(f"{save_path} is not exists.\n")
#             self._build_data_reocrd(self.args.model, save_path)
            
#         end_files = _implement_coverage_selection_single_process(
#                         save_path, 
#                         self.coverages, 
#                         self.args.choice_samples, 
#                         tmp_path = f"{path}/{self.args.model_name}"
#                     )
    
#         torch.save(end_files, f"{path}/{self.args.model_name}-end_files-ALL.pt")
#         return end_files

#     def _emergency_cleanup(self):
#         """紧急内存清理"""
#         self.batch_cache.clear()

#         # ✅ 清理原始属性以保持兼容性
#         self.layer_output_input_dict = {}
#         self.layer_dict_keys_cnt = 0

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()

#     def _cleanup_hooks(self):
#         """清理hooks"""
#         for handle in self.handles:
#             handle.remove()
#         self.handles.clear()
#         self._emergency_cleanup()



# def _process_single_coverage(args):
#     cov_name, covers, current_data_record, choice_samples = args
#     progress = tqdm(total=choice_samples,desc=f"  ->Coverage {cov_name} select files")
#     selected_files = set()
#     # 遍历每个覆盖度量实例
#     for cover_idx, cover in enumerate(covers):
#         while len(selected_files)<choice_samples:
#             cov_data_record = []
            
#             # 处理当前数据记录中的每个条目
#             for key, infos in current_data_record.items():
#                 for info in infos:
#                     try:
#                         output, path, seg_path = info
#                         _, cove_dict = cover.calculate(output)
#                         gain = cover.gain(cove_dict)

#                         cov_data_record.append([cove_dict, gain, path, seg_path])
#                     except Exception as e:
#                         print(f"    -> 处理数据 {key} 时出错: {e}")
#                         continue
            
#             max_cove_dict, max_gain, max_path, max_seg_path  = max(cov_data_record, key=lambda x: x[1])
            
#             # 如果gain <= 0，停止循环迭代
#             if max_gain <= 0:
#                 break
            
#             # 更新覆盖度量
#             cover.update(max_cove_dict, max_gain)
            
#             # 将选择的样本添加到结果中
#             selected_files.add((max_path, max_seg_path))
#             progress.update(1)

#         print(f"  -> {cov_name} 当前最大的 gain 为{max_gain}.当前覆盖率为： {cover.current}. 共有文件 {len(selected_files)}")

#         sorted_data = sorted(cov_data_record, key=lambda x: x[1], reverse=True)
#         for data in sorted_data:
#             gain, path, seg_path = data[1],data[2],data[3]
#             selected_files.add((path, seg_path))
#             if len(selected_files) >= choice_samples:
#                 break


#     return (cov_name, list(selected_files))


# def _implement_coverage_selection_single_process(save_path, coverages, choice_samples,tmp_path):
#     end_files = {}
#     current_data_record = torch.load(save_path)
    
#     for cov_name, covers in coverages.items():
#         for cover in covers:
#             if os.path.exists(f"{tmp_path}-end_files-{cov_name}.pt"):
#                 end_files[cov_name] = torch.load(f"{tmp_path}-end_files-{cov_name}.pt")
#                 print(f"  -> {cov_name} 已经存在，跳过处理, 有文件 {len(end_files[cov_name])} 个。")
#             _, selected_files = _process_single_coverage((cov_name, [cover], current_data_record, choice_samples))
#             end_files[cov_name] = selected_files
#             torch.save(end_files, f"{tmp_path}-end_files-{cov_name}.pt")
    
#     return end_files



# class CoverageAddHook(Hook):
#     """内存优化的覆盖率添加Hook"""
    
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
        
#         # 内存管理配置
#         self.batch_size = 1
#         self.memory_cleanup_interval = 100  # 更频繁的清理
#         self.max_memory_usage = 0.9  # 最大内存使用率80%
#         self.emergency_cleanup_threshold = 0.95  # 紧急清理阈值90%
        
#         # 基本配置
#         self.TOOL_LOG_FILE_PATH = args.TOOL_LOG_FILE_PATH
#         self.coverages = args.coverages
#         self.layer_dict = args.layer_dict
#         self.layer_dict_keys = list(args.output_sizes.keys())
#         self.file_name = args.config.split("/")[-1].replace(".py", "")
#         self.type = args.mode
#         self.output = None
        
#         # Hook管理
#         self.handles = []
        
#         # 优化的数据管理
#         self.current_batch_data = {}
#         self.processed_count = 0
        
#         # 临时文件管理
#         self.temp_dir = tempfile.mkdtemp(prefix="coverage_hook_")
#         self.temp_files = []
        
#         # 内存监控
#         self.memory_monitor = MemoryMonitor()
        
#         print(f"初始化CoverageAddHook，临时目录: {self.temp_dir}")

#     def _get_memory_usage(self):
#         """获取当前内存使用率"""
#         if torch.cuda.is_available():
#             gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
#             return max(gpu_memory, psutil.virtual_memory().percent / 100.0)
#         return psutil.virtual_memory().percent / 100.0

#     def _emergency_cleanup(self):
#         """紧急内存清理"""
#         # 清理当前批次数据
#         self.current_batch_data.clear()
        
#         # 强制垃圾回收
#         gc.collect()
        
#         # GPU内存清理
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
            

#     def _should_cleanup(self):
#         """检查是否需要清理内存"""
#         return (self.processed_count % self.memory_cleanup_interval == 0 or 
#                 self._get_memory_usage() > self.max_memory_usage)

#     def _build_data_evaluator(self):
#         """构建数据加载器"""
#         cfg = Config.fromfile(self.args.config)
        
#         dataloader = cfg.val_dataloader
#         dataloader.batch_size = self.batch_size
#         dataloader.num_workers = min(4, mp.cpu_count())  # 限制worker数量
        
#         if hasattr(dataloader, 'batch_sampler'):
#             dataloader.batch_sampler.batch_size = self.batch_size
            
#         dataloader = self.args.runner.build_dataloader(dataloader)
#         return dataloader

#     def _build_hooks(self):
#         """构建hooks"""
#         for name, module in self.layer_dict.items():
#             if isinstance(module, nn.Module):
#                 handle = module.register_forward_hook(self._create_hook_fn(name))
#                 self.handles.append(handle)

#     def _create_hook_fn(self, module_id):
#         """创建内存优化的hook函数"""
#         def hook(module, input, output):
#             # 检查内存使用情况
#             if self._get_memory_usage() > self.emergency_cleanup_threshold:
#                 self._emergency_cleanup()
#                 return  # 跳过当前处理以节省内存
            
#             # 使用更高效的tensor处理
#             with torch.no_grad():
#                 if output.size()[0] != 1:
#                     # 直接在原地操作，避免复制
#                     tmp_output = output.mean(dim=0, keepdim=True)
#                     if len(tmp_output.size()) > 2:
#                         tmp_output = tmp_output.mean(dim=tuple(range(2, len(tmp_output.size()))))
#                 else:
#                     # 避免clone，直接使用view
#                     tmp_output = output.detach()
#                     if len(tmp_output.size()) == 4 and not isinstance(module, nn.Linear):
#                         tmp_output = tmp_output.mean((2, 3))
#                     elif len(tmp_output.size()) == 4 and isinstance(module, nn.Linear):
#                         tmp_output = tmp_output.mean((1, 2))
#                     elif len(tmp_output.size()) == 3:
#                         tmp_output = tmp_output.mean(dim=1)
                
#                 # 立即转移到CPU以节省GPU内存
#                 tmp_output = tmp_output.cpu()
            
#             # 找到对应的layer名称
#             layer_name = None
#             for name, m in self.args.model.named_modules():
#                 if m is module:
#                     layer_name = f"{len(self.current_batch_data.get(self.input_name, {})) + 1}-{name}"
#                     break
            
#             if layer_name and layer_name in self.layer_dict_keys:
#                 if self.input_name not in self.current_batch_data:
#                     self.current_batch_data[self.input_name] = {}
                
#                 # 只保存必要的数据
#                 self.current_batch_data[self.input_name][layer_name] = tmp_output.clone()
                
#             # 立即清理临时变量
#             del tmp_output
            
#         return hook

#     def _save_batch_to_temp_file(self, batch_data, batch_idx):
#         """将批次数据保存到临时文件"""
#         temp_file = os.path.join(self.temp_dir, f"batch_{batch_idx}.pt")
        
#         # 压缩保存以节省空间
#         torch.save(batch_data, temp_file, pickle_protocol=4)
#         self.temp_files.append(temp_file)
        
#         return temp_file

#     def _load_batch_from_temp_file(self, temp_file):
#         """从临时文件加载批次数据"""
#         return torch.load(temp_file, map_location='cpu')

#     def _build_data_record(self, model):
#         """内存优化的数据记录构建"""
#         dataloader = self._build_data_evaluator()
#         batch_count = 0
        
#         progress = tqdm(total=len(dataloader), desc="处理数据批次（内存优化）")
        
#         with torch.no_grad():
#             for idx, data_batch in enumerate(dataloader):
#                 # 内存检查
#                 if self._get_memory_usage() > self.emergency_cleanup_threshold:
#                     self._emergency_cleanup()
                
#                 self.input_name = data_batch["data_samples"][0].img_path
                
#                 # 模型推理
#                 outputs = model.val_step(data_batch)
                
#                 # 获取路径信息
#                 path = self.input_name
#                 seg_path = data_batch['data_samples'][0].seg_map_path
                
#                 # 添加路径信息到当前批次数据
#                 if self.input_name in self.current_batch_data:
#                     self.current_batch_data[self.input_name]['_path_info'] = (path, seg_path)
                
#                 # 定期保存批次数据到临时文件
#                 if self._should_cleanup():  # 批次大小限制
#                     if self.current_batch_data:
#                         self._save_batch_to_temp_file(self.current_batch_data.copy(), batch_count)
#                         batch_count += 1
#                         self.current_batch_data.clear()
                    
#                     # 执行内存清理
#                     self._emergency_cleanup()
                
#                 # 清理当前循环的变量
#                 del outputs, data_batch
#                 self.processed_count += 1
#                 progress.update(1)
        
#         # 保存最后一批数据
#         if self.current_batch_data:
#             self._save_batch_to_temp_file(self.current_batch_data.copy(), batch_count)
#             self.current_batch_data.clear()
        
#         progress.close()
#         print(f"数据处理完成，共生成 {len(self.temp_files)} 个临时文件")

#     def _implement_coverage_selection(self, coverages, choice_samples, tmp_path):
#         """内存优化的覆盖率选择"""
        
#         re_files = set()
#         for cov_name, covers in coverages.items():
#             end_files = {}
#             print(f"处理覆盖率: {cov_name}")
            
#             # 检查是否已经存在结果
#             result_file = f"{tmp_path}-end_files-{cov_name}.pt"
#             if os.path.exists(result_file):
#                 end_files[cov_name] = torch.load(result_file)
#                 print(f"  -> {cov_name} 已存在，跳过处理，共 {len(end_files[cov_name])} 个文件")
#                 continue
            
#             selected_files = set()
            
#             for cover_idx, cover in enumerate(covers):
#                 progress = tqdm(total=choice_samples, desc=f"  -> 覆盖率 {cov_name} 选择文件")
                
#                 while len(selected_files) < choice_samples:
#                     best_gain = -1
#                     best_data = None
#                     current_processed = 0
                    
#                     # 遍历所有临时文件
#                     for temp_file in self.temp_files:
#                         try:
#                             # 分批加载数据
#                             batch_data = self._load_batch_from_temp_file(temp_file)
                            
#                             for key, data_info in batch_data.items():
#                                 if '_path_info' not in data_info:
#                                     continue
                                    
#                                 path, seg_path = data_info['_path_info']
                                
#                                 # 移除路径信息，只保留layer数据
#                                 layer_data = {k: v for k, v in data_info.items() if k != '_path_info'}
                                
#                                 try:
#                                     _, cove_dict = cover.calculate(layer_data)
#                                     gain = cover.gain(cove_dict)
                                    
#                                     if gain > best_gain:
#                                         best_gain = gain
#                                         best_data = (cove_dict, gain, path, seg_path)
                                        
#                                 except Exception as e:
#                                     print(f"计算覆盖率时出错: {e}")
#                                     continue
                            
#                             # 清理批次数据
#                             del batch_data
#                             current_processed += 1
                            
#                             # 定期清理内存
#                             if current_processed % 5 == 0:
#                                 gc.collect()
                                
#                         except Exception as e:
#                             print(f"加载临时文件 {temp_file} 时出错: {e}")
#                             continue
                    
#                     # 如果没有找到有效的数据或gain <= 0，停止
#                     if best_data is None or best_gain <= 0:
#                         break
                    
#                     # 更新覆盖率
#                     max_cove_dict, max_gain, max_path, max_seg_path = best_data
#                     cover.update(max_cove_dict, max_gain)
#                     selected_files.add((max_path, max_seg_path))
                    
#                     progress.update(1)
                    
#                     if len(selected_files) >= choice_samples:
#                         break
                
#                 progress.close()
#                 print(f"  -> {cov_name} 当前覆盖率: {cover.current}, 选择文件: {len(selected_files)}")
            
#             # 保存结果
#             end_files[cov_name] = list(selected_files)
#             torch.save(end_files[cov_name], result_file)
#             re_files.add(end_files[cov_name])

#         return list(re_files)

#     def __call__(self):
#         """主执行函数"""
#         try:
#             self._build_hooks()
#             self.args.model.eval()
            
#             # 设置路径
#             path = f"{self.args.TOOL_LOG_FILE_PATH}/cov_record/"
#             utility.build_path(path)
#             save_path = f"{path}/{self.args.model_name}_coverage_info.pt"
            
#             # 构建数据记录（内存优化版本）
#             print(f"开始构建数据记录...")
#             self._build_data_record(self.args.model)
            
#             # 执行覆盖率选择（内存优化版本）
#             print(f"开始执行覆盖率选择...")
#             end_files = self._implement_coverage_selection(
#                 self.coverages,
#                 self.args.choice_samples,
#                 tmp_path=f"{path}/{self.args.model_name}"
#             )
            
#             # 保存最终结果
#             final_result_path = f"{path}/{self.args.model_name}-end_files-ALL.pt"
#             torch.save(end_files, final_result_path)
#             print(f"结果已保存到: {final_result_path}")
            
#             return end_files
            
#         finally:
#             self._cleanup()

#     def _cleanup(self):
#         """清理资源"""
#         # 移除hooks
#         for handle in self.handles:
#             try:
#                 handle.remove()
#             except:
#                 pass
#         self.handles.clear()
        
#         # 清理内存
#         self.current_batch_data.clear()
        
#         # 删除临时文件
#         try:
#             if os.path.exists(self.temp_dir):
#                 shutil.rmtree(self.temp_dir)
#                 print(f"已清理临时目录: {self.temp_dir}")
#         except Exception as e:
#             print(f"清理临时目录时出错: {e}")
        
#         # 最终内存清理
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.synchronize()
        
#         print("资源清理完成")




class CoverageAddHook(Hook):
    """内存优化的覆盖率添加Hook（永久文件版本）"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 内存管理配置
        self.batch_size = 1
        self.memory_cleanup_interval = 100  # 更频繁的清理
        self.max_memory_usage = 0.9  # 最大内存使用率80%
        self.emergency_cleanup_threshold = 0.95  # 紧急清理阈值90%
        
        # 基本配置
        self.TOOL_LOG_FILE_PATH = args.TOOL_LOG_FILE_PATH
        self.coverages = args.coverages
        self.layer_dict = args.layer_dict
        self.layer_dict_keys = list(args.output_sizes.keys())
        self.file_name = args.config.split("/")[-1].replace(".py", "")
        self.type = args.mode
        self.output = None
        
        # Hook管理
        self.handles = []
        
        # 优化的数据管理
        self.current_batch_data = {}
        self.processed_count = 0
        
        # 永久文件管理
        self.permanent_dir = os.path.join(self.TOOL_LOG_FILE_PATH, "cov_record", "data_cache", self.args.model_name)
        os.makedirs(self.permanent_dir, exist_ok=True)
        self.permanent_files = []
        
        # 内存监控
        self.memory_monitor = MemoryMonitor()
        
        print(f"初始化CoverageAddHook，永久数据目录: {self.permanent_dir}")

    def _get_memory_usage(self):
        """获取当前内存使用率"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return max(gpu_memory, psutil.virtual_memory().percent / 100.0)
        return psutil.virtual_memory().percent / 100.0

    def _emergency_cleanup(self):
        """紧急内存清理"""
        # 清理当前批次数据
        self.current_batch_data.clear()
        
        # 强制垃圾回收
        gc.collect()
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            

    def _should_cleanup(self):
        """检查是否需要清理内存"""
        return (self.processed_count % self.memory_cleanup_interval == 0 or 
                self._get_memory_usage() > self.max_memory_usage)

    def _build_data_evaluator(self):
        """构建数据加载器"""
        cfg = Config.fromfile(self.args.config)
        
        dataloader = cfg.val_dataloader
        dataloader.batch_size = self.batch_size
        dataloader.num_workers = min(4, mp.cpu_count())  # 限制worker数量
        
        if hasattr(dataloader, 'batch_sampler'):
            dataloader.batch_sampler.batch_size = self.batch_size
            
        dataloader = self.args.runner.build_dataloader(dataloader)
        return dataloader

    def _build_hooks(self):
        """构建hooks"""
        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):
                handle = module.register_forward_hook(self._create_hook_fn(name))
                self.handles.append(handle)

    def _create_hook_fn(self, module_id):
        """创建内存优化的hook函数"""
        def hook(module, input, output):
            # 检查内存使用情况
            if self._get_memory_usage() > self.emergency_cleanup_threshold:
                self._emergency_cleanup()
                return  # 跳过当前处理以节省内存
            
            # 使用更高效的tensor处理
            with torch.no_grad():
                if output.size()[0] != 1:
                    # 直接在原地操作，避免复制
                    tmp_output = output.mean(dim=0, keepdim=True)
                    if len(tmp_output.size()) > 2:
                        tmp_output = tmp_output.mean(dim=tuple(range(2, len(tmp_output.size()))))
                else:
                    # 避免clone，直接使用view
                    tmp_output = output.detach()
                    if len(tmp_output.size()) == 4 and not isinstance(module, nn.Linear):
                        tmp_output = tmp_output.mean((2, 3))
                    elif len(tmp_output.size()) == 4 and isinstance(module, nn.Linear):
                        tmp_output = tmp_output.mean((1, 2))
                    elif len(tmp_output.size()) == 3:
                        tmp_output = tmp_output.mean(dim=1)
                
                # 保存设备信息和数据（暂不转移到CPU）
                device_info = tmp_output.device
            
            # 找到对应的layer名称
            layer_name = None
            for name, m in self.args.model.named_modules():
                if m is module:
                    layer_name = f"{len(self.current_batch_data.get(self.input_name, {})) + 1}-{name}"
                    break
            
            if layer_name and layer_name in self.layer_dict_keys:
                if self.input_name not in self.current_batch_data:
                    self.current_batch_data[self.input_name] = {}
                
                # 保存数据和设备信息
                self.current_batch_data[self.input_name][layer_name] = {
                    'data': tmp_output.cpu().clone(),  # 转CPU保存以节省GPU内存
                    'device': str(device_info)  # 保存原始设备信息
                }
                
            # 立即清理临时变量
            del tmp_output
            
        return hook

    def _save_batch_to_permanent_file(self, batch_data, batch_idx):
        """将批次数据保存到永久文件"""
        permanent_file = os.path.join(self.permanent_dir, f"batch_{batch_idx}.pt")
        
        # 压缩保存以节省空间
        torch.save(batch_data, permanent_file, pickle_protocol=4)
        self.permanent_files.append(permanent_file)
        
        return permanent_file

    def _load_batch_from_permanent_file(self, permanent_file):
        """从永久文件加载批次数据"""
        return torch.load(permanent_file, map_location='cpu')

    def _check_existing_data_cache(self):
        """检查是否已存在数据缓存"""
        # 检查永久目录中是否已有数据文件
        if os.path.exists(self.permanent_dir):
            existing_files = [f for f in os.listdir(self.permanent_dir) if f.startswith('batch_') and f.endswith('.pt')]
            if existing_files:
                # 按文件名排序确保顺序正确
                existing_files.sort(key=lambda x: int(x.replace('batch_', '').replace('.pt', '')))
                self.permanent_files = [os.path.join(self.permanent_dir, f) for f in existing_files]
                print(f"发现已存在的数据缓存，共 {len(self.permanent_files)} 个文件")
                return True
        return False

    def _build_data_record(self, model):
        """内存优化的数据记录构建（支持永久文件缓存）"""
        # 首先检查是否已存在数据缓存
        if self._check_existing_data_cache():
            print("使用已存在的数据缓存，跳过数据记录构建")
            return
        
        print("未发现数据缓存，开始构建新的数据记录...")
        
        dataloader = self._build_data_evaluator()
        batch_count = 0
        
        progress = tqdm(total=len(dataloader), desc="处理数据批次（永久文件版本）")
        
        with torch.no_grad():
            for idx, data_batch in enumerate(dataloader):
                # 内存检查
                if self._get_memory_usage() > self.emergency_cleanup_threshold:
                    self._emergency_cleanup()
                
                self.input_name = data_batch["data_samples"][0].img_path
                
                # 模型推理
                outputs = model.val_step(data_batch)
                
                # 获取路径信息
                path = self.input_name
                seg_path = data_batch['data_samples'][0].seg_map_path
                
                # 添加路径信息到当前批次数据
                if self.input_name in self.current_batch_data:
                    self.current_batch_data[self.input_name]['_path_info'] = (path, seg_path)
                
                # 定期保存批次数据到永久文件
                if self._should_cleanup():  # 批次大小限制
                    if self.current_batch_data:
                        self._save_batch_to_permanent_file(self.current_batch_data.copy(), batch_count)
                        batch_count += 1
                        self.current_batch_data.clear()
                    
                    # 执行内存清理
                    self._emergency_cleanup()
                
                # 清理当前循环的变量
                del outputs, data_batch
                self.processed_count += 1
                progress.update(1)
        
        # 保存最后一批数据
        if self.current_batch_data:
            self._save_batch_to_permanent_file(self.current_batch_data.copy(), batch_count)
            self.current_batch_data.clear()
        
        progress.close()
        print(f"数据记录构建完成，共生成 {len(self.permanent_files)} 个永久文件")

    def _implement_coverage_selection(self, coverages, choice_samples, tmp_path):
        """内存优化的覆盖率选择"""
        
        # 检测计算设备
        compute_device = next(self.args.model.parameters()).device
        print(f"使用设备进行覆盖率计算: {compute_device}")
        
        re_files = {}
        for cov_name, covers in coverages.items():
            end_files = {}
            prep_cnt = 0
            print(f"处理覆盖率: {cov_name}")
            
            # 检查是否已经存在结果
            result_file = f"{tmp_path}-end_files-{cov_name}.pt"
            if os.path.exists(result_file):
                end_files[cov_name] = torch.load(result_file)
                print(f"  -> {cov_name} 已存在，跳过处理，共 {len(end_files[cov_name])} 个文件")
                continue
            
            selected_files = set()
            
            for cover_idx, cover in enumerate(covers):
                progress = tqdm(total=choice_samples, desc=f"  -> 覆盖率 {cov_name} 选择文件")
                
                while len(selected_files) < choice_samples:
                    best_gain = -1
                    best_data = None
                    current_processed = 0

                    if cov_name in ["CC", "NLC"]:
                        perpare_cov = copy.deepcopy(cover)
                        if prep_cnt == 0:
                            perpare_data = self.permanent_files[0]
                            perpare_data = self._load_batch_from_permanent_file(perpare_data)
                            layer_data = {}
                            for _, data_info in perpare_data.items():
                                layer_data = {}
                                for k, v in data_info.items():
                                    if k != '_path_info':
                                        if isinstance(v, dict) and 'data' in v:
                                            # 新格式：包含设备信息的数据
                                            tensor_data = v['data'].to(compute_device)
                                            layer_data[k] = tensor_data
                                        else:
                                            # 兼容旧格式：直接是tensor
                                            if isinstance(v, torch.Tensor):
                                                layer_data[k] = v.to(compute_device)
                                            else:
                                                layer_data[k] = v

                            _, cove_dict = cover.calculate(layer_data)
                            gain = cover.gain(cove_dict)
                            cover.update(cove_dict, gain)
                            prep_cnt = 1


                    # 遍历所有永久文件
                    for permanent_file in self.permanent_files:
                        try:
                            # 分批加载数据
                            batch_data = self._load_batch_from_permanent_file(permanent_file)
                            
                            for key, data_info in batch_data.items():
                                if '_path_info' not in data_info:
                                    continue
                                    
                                path, seg_path = data_info['_path_info']
                                
                                # 移除路径信息，准备layer数据
                                layer_data = {}
                                for k, v in data_info.items():
                                    if k != '_path_info':
                                        if isinstance(v, dict) and 'data' in v:
                                            # 新格式：包含设备信息的数据
                                            tensor_data = v['data'].to(compute_device)
                                            layer_data[k] = tensor_data
                                        else:
                                            # 兼容旧格式：直接是tensor
                                            if isinstance(v, torch.Tensor):
                                                layer_data[k] = v.to(compute_device)
                                            else:
                                                layer_data[k] = v

                                try:
                                    _, cove_dict = cover.calculate(layer_data)
                                    gain = cover.gain(cove_dict)
                                    
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_data = (cove_dict, gain, path, seg_path)
                                        
                                except Exception as e:
                                    print(f"计算覆盖率时出错: {e}")
                                    continue
                                finally:
                                    # 清理GPU内存
                                    del layer_data

                            # 清理批次数据
                            del batch_data
                            current_processed += 1
                            
                            # 定期清理内存
                            if current_processed % 5 == 0:
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                        except Exception as e:
                            print(f"加载永久文件 {permanent_file} 时出错: {e}")
                            continue
                    
                    # 如果没有找到有效的数据或gain <= 0，停止
                    if best_data is None or best_gain <= 0:
                        break
                    
                    # 更新覆盖率
                    if prep_cnt == 1:
                        cover = perpare_cov

                    max_cove_dict, max_gain, max_path, max_seg_path = best_data
                    cover.update(max_cove_dict, max_gain)
                    selected_files.add((max_path, max_seg_path))


                    progress.update(1)
                    
                    if len(selected_files) >= choice_samples:
                        break
                
                progress.close()
                print(f"  -> {cov_name} 当前覆盖率: {cover.current}, 选择文件: {len(selected_files)}")
            
            # 保存结果
            end_files[cov_name] = list(selected_files)
            torch.save(end_files, result_file)
            re_files.update(end_files)

        return list(re_files)

    def clear_data_cache(self):
        """清理数据缓存（手动调用）"""
        try:
            if os.path.exists(self.permanent_dir):
                shutil.rmtree(self.permanent_dir)
                os.makedirs(self.permanent_dir, exist_ok=True)
                print(f"已清理数据缓存目录: {self.permanent_dir}")
                self.permanent_files.clear()
        except Exception as e:
            print(f"清理数据缓存时出错: {e}")

    def __call__(self):
        """主执行函数"""
        try:
            self._build_hooks()
            self.args.model.eval()
            
            # 设置路径
            path = f"{self.args.TOOL_LOG_FILE_PATH}/cov_record/"
            utility.build_path(path)
            save_path = f"{path}/{self.args.model_name}_coverage_info.pt"
            
            # 构建数据记录（支持永久文件缓存）
            print(f"开始构建数据记录...")
            self._build_data_record(self.args.model)
            
            # 执行覆盖率选择（使用永久文件）
            print(f"开始执行覆盖率选择...")
            end_files = self._implement_coverage_selection(
                self.coverages,
                self.args.choice_samples,
                tmp_path=f"{path}/{self.args.model_name}"
            )
            
            # 保存最终结果
            final_result_path = f"{path}/{self.args.model_name}-end_files-ALL.pt"
            torch.save(end_files, final_result_path)
            print(f"结果已保存到: {final_result_path}")
            
            return end_files
            
        finally:
            self._cleanup()

    def _cleanup(self):
        """清理资源（保留永久文件）"""
        # 移除hooks
        for handle in self.handles:
            try:
                handle.remove()
            except:
                pass
        self.handles.clear()
        
        # 清理内存
        self.current_batch_data.clear()
        
        # 注意：不删除永久文件目录，保留数据缓存
        print(f"保留数据缓存目录: {self.permanent_dir}")
        
        # 最终内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("资源清理完成（数据缓存已保留）")



















class MemoryMonitor:
    """内存监控类"""
    
    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控内存"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控内存"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU内存
                cpu_percent = psutil.virtual_memory().percent
                
                # GPU内存
                gpu_info = ""
                if torch.cuda.is_available():
                    gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                    gpu_info = f", GPU: {gpu_allocated:.1f}GB allocated, {gpu_cached:.1f}GB cached"
                
                if cpu_percent > 90:
                    print(f"⚠️  内存警告: CPU {cpu_percent:.1f}%{gpu_info}")
                    
            except Exception as e:
                print(f"内存监控出错: {e}")
            
            time.sleep(self.check_interval)


# 使用示例和额外的优化函数
def optimize_torch_settings():
    """优化PyTorch设置以减少内存使用"""
    # 设置内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 启用内存管理优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # 减少内存碎片
        torch.backends.cudnn.deterministic = True
        
    # 设置线程数
    torch.set_num_threads(min(4, mp.cpu_count()))
    
    print("PyTorch内存优化设置已应用")


def monitor_memory_usage(func):
    """内存使用监控装饰器"""
    def wrapper(*args, **kwargs):
        # 记录开始时的内存使用
        start_memory = psutil.virtual_memory().percent
        if torch.cuda.is_available():
            start_gpu = torch.cuda.memory_allocated() / 1024**3
            
        print(f"函数开始执行，内存使用: CPU {start_memory:.1f}%", end="")
        if torch.cuda.is_available():
            print(f", GPU {start_gpu:.1f}GB")
        else:
            print()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 记录结束时的内存使用
            end_memory = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                end_gpu = torch.cuda.memory_allocated() / 1024**3
                print(f"函数执行完成，内存使用: CPU {end_memory:.1f}% (Δ{end_memory-start_memory:+.1f}%), GPU {end_gpu:.1f}GB (Δ{end_gpu-start_gpu:+.1f}GB)")
            else:
                print(f"函数执行完成，内存使用: CPU {end_memory:.1f}% (Δ{end_memory-start_memory:+.1f}%)")
    
    return wrapper