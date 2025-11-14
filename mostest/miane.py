



"""
MOSTest 主执行文件
MOSTest Main Execution File
"""
import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import gc
import warnings
from torch.utils.data import DataLoader
import threading
import coverage
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from mmseg.apis import inference_model, init_model
from utility import *
import copy
import uuid
import json
import time
import torch
import torch.nn as nn
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import mmcv
from multiprocessing import Pool
from rich.console import Console

try:
    from .core.chromosome import Chromosome, MyDataset
    from .core.transform_registry import TransformRegistry, TRANSFORM_CONFIGS, TransformConfig
    from .objectives.f1_neural_behavior import F1NeuralBehavior
    from .objectives import F2SemanticQuality, F3FeatureConsistency
    from .optimization.nsga3 import NSGA3
    from .metrics.boundary import BoundaryCoverage
    from .mostconfig import MOSTestConfig
except (ImportError, ValueError):
    from core.chromosome import Chromosome, MyDataset
    from core.transform_registry import TransformRegistry, TRANSFORM_CONFIGS, TransformConfig
    from objectives.f1_neural_behavior import F1NeuralBehavior
    from objectives import F2SemanticQuality, F3FeatureConsistency
    from optimization.nsga3 import NSGA3
    from metrics.boundary import BoundaryCoverage
    from mostconfig import MOSTestConfig

console = Console()
warnings.filterwarnings('ignore', message=".*force_all_finite.*", category=FutureWarning)


def get_model_layers(model):
    name_counter = {}
    layer_dict = {}
    cnt = 0

    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [], "")
        assert len(name_list) == len(module_list)

        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__

            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1

            layer_dict['%d-%d-%s-%s' % (cnt, name_counter[class_name], name_list[i], class_name)] = module
            cnt += 1
    return layer_dict

def evaluate_chromosome_func(datas):
    # 计算F1: 神经行为覆盖 Compute F1: Neural behavior coverage
    func1, func2, func3 = datas["func1"], datas["func2"], datas["func3"]
    del datas["func1"], datas["func2"], datas["func3"]

    f1_value = func1.compute(**datas)
    f2_value = func2.compute(**datas)
    f3_value = func3.compute(**datas)

    return [f1_value, f2_value, f3_value]

def _process_chromosome_mutation(args):
    idx, chromosome, seed_idx, file_name, seed_image, seed_mask = args

    try:
        transform, mutated_image, mutated_mask = chromosome.apply_transform(seed_image, seed_mask)

        finames = file_name.split("_")
        finames = f"{finames[0]}_{str(transform[0]).split('(')[0]}_{str(uuid.uuid4())[:8]}_{'_'.join(finames[1:])}"

        muta_data = {
            "seed_idx": seed_idx,
            "uuid":str(uuid.uuid4())[:8],
            "muta_image": mutated_image,
            "muta_mask": mutated_mask,
            "file_name": finames,
        }

        if np.all(mutated_image == seed_image) or np.all(mutated_image == 0):
            return (idx, False, None, None, None, None)

        return (idx, True, muta_data, file_name, seed_idx, mutated_image)
    except Exception as e:
        print(f"Error processing chromosome {idx}: {e}")
        return (idx, False, None, None, None, None)


class MOSTest:
    """
    Multi-Objective Semantic Segmentation Testing Framework
    """

    def __init__(
            self,
            mostest_config: Optional[MOSTestConfig] = None  # 新增：配置对象
    ):
        assert mostest_config is not None, "Must provide MOSTestConfig object."
        mostest_config.set_random_seed()
        self.model_name = mostest_config.model_name
        self.dataset = mostest_config.dataset
        self.model_type = mostest_config.model_type
        self.config = mostest_config.config_file
        self.checkpoint = mostest_config.checkpoint_file
        self.device = mostest_config.device
        self.model = init_model(self.config, self.checkpoint, device=self.device)
        self.model.eval()

        now = datetime.now()
        day = now.strftime("%m%d")
        # self.output_dir = mostest_config.output_dir or Path(
        #     f"./mostest_output/output-data-{day}/{self.dataset}/{self.model_type}/{self.model_name}")
        self.output_dir = mostest_config.output_dir or Path(
            f"./mostest_output/output-data-None/{self.dataset}/{self.model_type}/{self.model_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coverage_save_path = self.output_dir / 'coverage'
        self.coverage_save_path.mkdir(parents=True, exist_ok=True)
        self.TOOL_LOG_FILE_PATH = f"../temp/{self.dataset}/{self.model_type}/{self.model_name}"
        self.generations_dir = self.output_dir / 'generations'
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.pareto_fronts_dir = self.output_dir / 'pareto_fronts'
        self.pareto_fronts_dir.mkdir(parents=True, exist_ok=True)

        top_k = mostest_config.top_k
        # Use CPU-optimized F1 for better multiprocessing performance
        self.f1 = F1NeuralBehavior(top_k=top_k)
        self.f2 = F2SemanticQuality()
        self.f3 = F3FeatureConsistency()

        self.save_output = None
        self.layer_output_input_dict = {}
        self.handles = []
        self.layer_dict = get_model_layers(self.model)
        self.layer_dict_keys = list(self.layer_dict.keys())
        self.output_sizes = {}
        self._get_output_size()
        self.layer_dict_keys_cnt = 0
        self.handles = []
        self.register_hooks()

        self.eval_lock = threading.Lock()

        self.ori_seed_prediction_cache = {}
        self.muta_seed_prediction_cache = {}

        self.num_workers = mostest_config.num_workers
        self.optimizer = NSGA3(mostest_config=mostest_config)
        self.max_runtime_hours = mostest_config.max_runtime_hours

        self.tknp_coverage = getattr(coverage, "TKNP")(
            model_name=self.model_name,
            layer_size_dict=self.output_sizes,
            device=self.device,
            save_path=str(self.coverage_save_path),
            TOOL_LOG_FILE_PATH=self.TOOL_LOG_FILE_PATH,
            threshold=top_k
        )

        self.num_classes = mostest_config.num_classes
        self.boundary_coverage = BoundaryCoverage(self.num_classes)
        self.activation_coverage = getattr(coverage, "ADC")(
            model_name=self.model_name,
            layer_size_dict=self.output_sizes,
            device=self.device,
            save_path=str(self.coverage_save_path),
            TOOL_LOG_FILE_PATH=self.TOOL_LOG_FILE_PATH,
            threshold=mostest_config.num_bins
        )

        self.transform_sequence = mostest_config.transform_sequence or TRANSFORM_CONFIGS
        self.mostest_config = mostest_config

        self.history = {
            'generations': [],
            'pareto_fronts': [],
            'coverage_history': {
                'tknp': [],
                'boundary': [],
                'activation': []
            }
        }

    def _get_output_size(self) -> tuple:

        def _size_hook_fun(layer_name):
            def hook(module, input, output):
                for name, m in self.model.named_modules():
                    if m is module:
                        name = f"{self.layer_dict_keys_cnt + 1}-{name}"
                        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or
                                isinstance(m, nn.Conv3d) or isinstance(m, nn.GroupNorm)):
                            out_size = output.size()[1]
                        elif isinstance(m, nn.Linear):
                            out_size = m.out_features
                        elif isinstance(m, nn.LayerNorm):
                            out_size = m.normalized_shape[0]
                        else:
                            console.print(f"[yellow]Warning: Unknown module type: {type(m).__name__}[/yellow]")
                        self.output_sizes[name] = {"Output": [out_size], "Input": input[0].size()}
                        self.layer_dict_keys_cnt += 1
                        break

            return hook

        path = self.TOOL_LOG_FILE_PATH + "/output_sizes.pth"

        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):  # 只对 nn.Module 注册 hook
                handle = module.register_forward_hook(_size_hook_fun(name))
                self.handles.append(handle)

        if self.dataset == "ade20k":
            img = mmcv.imread("../demo/demo-ade20k.jpg")
        elif self.dataset == "cityscapes":
            img = mmcv.imread("../demo/demo.png")

        self.layer_dict_keys_cnt = 0
        inference_model(self.model, img)
        torch.save(self.output_sizes, path)
        self.layer_dict_keys_cnt = 0
        self.layer_dict_keys = list(self.output_sizes.keys())

        # 在运行结束后移除所有 hook
        for handle in self.handles:
            handle.remove()

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            assert len(input[0]) == output.shape[0], f"input has {len(input[0])}, output has {output.shape}."

            if output.size()[0] != 1:
                tmp_output = output.unsqueeze(0)
                tmp_output = tmp_output.mean((1, 2))
            else:
                tmp_output = output.detach()

                if len(tmp_output.size()) == 4 and not (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((2, 3))
                elif len(tmp_output.size()) == 4 and (isinstance(module, nn.Linear)):  # (N, K, H, w)
                    tmp_output = tmp_output.mean((1, 2))
                elif len(tmp_output.size()) == 3:  # (H, W, C)
                    tmp_output = tmp_output.mean(dim=1)
                else:
                    console.print(f"[red]Error: Unexpected output shape[/red]")
                    console.print(f"Module: {module}")
                    console.print(f"Output size: {tmp_output.size()}")
                    console.print(f"Dimensions: {len(tmp_output.size())}")
                    console.print(f"Is Linear: {isinstance(module, nn.Linear)}")
                    exit()

            for name, m in self.model.named_modules():
                if m is module:
                    name = f"{self.layer_dict_keys_cnt + 1}-{name}"
                    try:
                        if name == self.layer_dict_keys[self.layer_dict_keys_cnt]:
                            self.layer_output_input_dict[
                                self.layer_dict_keys[
                                    self.layer_dict_keys_cnt]] = tmp_output
                            break

                    except Exception as e:
                        console.print(f"[yellow]Warning: {name} not in layer_dict_keys: {e}. Skipping.[/yellow]")

            if self.output == None or not torch.equal(self.output, tmp_output):
                self.layer_dict_keys_cnt += 1

            self.output = tmp_output

        return hook

    def register_hooks(self):
        for name, module in self.layer_dict.items():
            if isinstance(module, nn.Module):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def remove_hooks(self):
        self.layer_output_input_dict = {}
        self.layer_dict_keys_cnt = 0
        self.save_output = None
        self.handles = []

        for handle in self.handles:
            handle.remove()

    def _evaluate_chromosome(self, population: List[Chromosome], datalist: Dict, desc: str) -> Chromosome:
        fault_cnt = 0
        pbar = tqdm(total=len(population), desc=desc, leave=True)

        def update(*args):
            pbar.update()

        use_dataloader_list = []
        for chromosome in tqdm(population, desc=desc + " [1] get mutated data", leave=True):
            seed_idx = chromosome.get_image_index()
            file_name, seed_image, seed_mask = datalist[seed_idx]

            transform, mutated_image, mutated_mask = chromosome.apply_transform(seed_image, seed_mask)

            finames = file_name.split("_")
            uuids = str(uuid.uuid4())[:8]
            finames = f"{finames[0]}_{str(transform[0]).split('(')[0]}_{uuids}_{'_'.join(finames[1:])}"
            muta_data = {
                "seed_idx": seed_idx,
                "uuid":uuids,
                "muta_image": mutated_image,
                "muta_mask": mutated_mask,
                "file_name": finames,
            }

            if np.all(mutated_image == seed_image) or np.all(mutated_image == 0):
                fault_cnt += 1
                chromosome.muta_data = None
                chromosome.objectives = [0.0, 0.0, 0.0]
                update()
                continue

            chromosome.ori_file_name = file_name
            chromosome.muta_data = muta_data

            use_dataloader_list.append((uuids, mutated_image))

        self.build_muta_seed_prediction_cache(use_dataloader_list,desc=desc+" [2] build muta seed prediction cache")

        with Pool(processes=self.num_workers) as pool:
            for chromosome_idxs in range(0, len(population), self.num_workers):
                results = []
                for chromosome_idx in range(chromosome_idxs, min(chromosome_idxs + self.num_workers, len(population))):
                    chromosome = population[chromosome_idx]
                    if chromosome.muta_data is None:
                        continue
                    seed_idx = chromosome.get_image_index()

                    ori_layer_output_dict, ori_pred_sem_seg = self.get_prediction_with_cache(
                        seed_idx, self.ori_seed_prediction_cache)
                    muta_layer_output_dict, muta_pred_sem_seg = self.get_prediction_with_cache(
                        chromosome.muta_data["uuid"], self.muta_seed_prediction_cache)

                    file_name, seed_image, seed_mask = datalist[seed_idx]
                    mutated_mask = chromosome.muta_data["muta_mask"]

                    data = {
                        "chromosome_idx": chromosome_idx,
                        "func1": self.f1,
                        "func2": self.f2,
                        "func3": self.f3,
                        "seed_idx": seed_idx,
                        "seed_mask": seed_mask,
                        "muta_mask": mutated_mask,
                        "ori_layer_output_dict": ori_layer_output_dict,
                        "ori_pred_sem_seg": ori_pred_sem_seg,
                        "muta_layer_output_dict": muta_layer_output_dict,
                        "muta_pred_sem_seg": muta_pred_sem_seg,
                        "num_classes": self.num_classes
                    }

                    result = pool.apply_async(evaluate_chromosome_func, args=(data,), callback=update)
                    results.append((chromosome_idx, result))

                for chromosome_idx, r in results:
                    chromosome = population[chromosome_idx]
                    f1_value, f2_value, f3_value = r.get()
                    if f1_value == 0.0 or f2_value == 0.0 or f3_value == 0.0:
                        fault_cnt += 1
                    chromosome.objectives = [f1_value, f2_value, f3_value]

                del results
                gc.collect()

            if fault_cnt > 0:
                console.print(
                    f"\n  [yellow]⚠[/yellow] The number of population is {len(population)} Removed: [yellow]{fault_cnt}[/yellow] invalid individuals")

        return population

    def _evaluate_population(self, population: List[Chromosome], datalist: Dict, desc: str = "evaluation") -> List[Chromosome]:
        population = self._evaluate_chromosome(population, datalist, desc)

        gc.collect()
        torch.cuda.empty_cache()
        return [pop for pop in population if pop.objectives != [0.0, 0.0, 0.0]]

    def update_coverage(self, population: List[Chromosome], desc: str = "Update coverage"):

        coverage_lock = threading.Lock()

        def process_single_chromosome(chromosome):
            try:
                muta_data = chromosome.muta_data
                if muta_data is None:
                    print("Muta data is None, skipping coverage update.")
                with coverage_lock:
                    muta_layer_output_dict, _ = self.get_prediction_with_cache(muta_data["uuid"], self.muta_seed_prediction_cache)
                    self.tknp_coverage.step(muta_layer_output_dict, muta_data["file_name"])
                    self.boundary_coverage.update(muta_data["muta_mask"])
                    self.activation_coverage.step(muta_layer_output_dict, muta_data["file_name"])

                return True, None

            except Exception as e:
                return False, str(e)

        with ThreadPoolExecutor(max_workers=40) as executor:
            future_to_idx = {
                executor.submit(process_single_chromosome, chrom): idx
                for idx, chrom in enumerate(population)
            }

            failed_count = 0
            with tqdm(total=len(population), desc=desc) as pbar:
                for future in as_completed(future_to_idx):
                    success, error_msg = future.result()
                    if not success:
                        failed_count += 1
                        idx = future_to_idx[future]
                        console.print(f"  [yellow]⚠[/yellow] Chromosome {idx} failed: [dim]{error_msg}[/dim]")
                    pbar.update(1)

            if failed_count > 0:
                console.print(f"  [yellow]⚠[/yellow] {failed_count}/{len(population)} chromosomes failed")

    def save_generation_data(self, generation: int, population: List[Chromosome], pareto_front: List[Chromosome] = None,
                             save_images: bool = True):
        gen_dir = self.generations_dir / f'gen/gen_{generation:03d}'
        pareto_dir = self.generations_dir / 'pareto_front'
        gen_dir.mkdir(parents=True, exist_ok=True)
        pareto_dir.mkdir(parents=True, exist_ok=True)

        pareto_file_names = set()
        if pareto_front:
            for ind in pareto_front:
                if ind.muta_data:
                    pareto_file_names.add(ind.muta_data['file_name'])

        if save_images:

            images_dir = gen_dir / 'cityscapes/leftImg8bit'
            masks_dir = gen_dir / 'cityscapes/gtFine'
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            pareto_images_dir = pareto_dir / 'cityscapes/leftImg8bit'
            pareto_masks_dir = pareto_dir / 'cityscapes/gtFine'
            pareto_images_dir.mkdir(parents=True, exist_ok=True)
            pareto_masks_dir.mkdir(parents=True, exist_ok=True)

            for chromosome in population:
                if chromosome.muta_data:
                    file_name = chromosome.muta_data['file_name']
                    muta_image = chromosome.muta_data['muta_image']
                    muta_mask = chromosome.muta_data['muta_mask']

                    cv2.imwrite(
                        str(images_dir / f"{file_name}"),
                        cv2.cvtColor(muta_image, cv2.COLOR_RGB2BGR)
                    )

                    mask_file_name = file_name.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
                    cv2.imwrite(
                        str(masks_dir / mask_file_name),
                        muta_mask
                    )

                    if file_name in pareto_file_names:
                        # For pareto front, save as jpg but need to strip .png first
                        base_name = file_name.replace('.png', '')
                        cv2.imwrite(
                            str(pareto_images_dir / f"{base_name}.jpg"),
                            cv2.cvtColor(muta_image, cv2.COLOR_RGB2BGR)
                        )
                        cv2.imwrite(
                            str(pareto_masks_dir / f"{file_name}"),
                            muta_mask
                        )

        fronts = self.optimizer.fast_non_dominated_sort(population)
        front_levels = {}
        for level, front in enumerate(fronts):
            for ind in front:
                if ind.muta_data:
                    front_levels[ind.muta_data['file_name']] = level

        population_data = []
        for idx, chromosome in enumerate(population):
            if not chromosome.muta_data:
                continue
            file_name = chromosome.muta_data['file_name']
            ind_data = {
                'index': idx,
                'file_name': file_name,
                'orig_file_name': chromosome.ori_file_name,

                'objectives': {
                    'F1_neural_behavior': chromosome.objectives[0],
                    'F2_semantic_quality': chromosome.objectives[1],
                    'F3_feature_consistency': chromosome.objectives[2]
                },

                'quality': chromosome.muta_data.get('quality', {}),

                'chromosome_params': {
                    'params': chromosome.params,
                    'enabled_transforms': chromosome.enabled_transforms
                },

                'status': {
                    'in_pareto_front': file_name in pareto_file_names,
                    'front_level': front_levels.get(file_name, -1),
                    'kept_for_next_gen': True,
                    'used_for_coverage_update': True
                }
            }

            population_data.append(ind_data)

        with open(gen_dir / 'population.json', 'w', encoding='utf-8') as f:
            json.dump(population_data, f, indent=2, ensure_ascii=False)

        status_data = {
            'generation': generation,
            'population_size': len(population),
            'pareto_front_size': len(pareto_file_names),
            'front_distribution': {
                f'front_{i}': len(front) for i, front in enumerate(fronts)
            },
            'coverage': {
                'tknp': self.tknp_coverage.current,
                'boundary': self.boundary_coverage.get_coverage(),
                'activation': self.activation_coverage.current
            },
            'objective_statistics': {
                'F1': {
                    'mean': float(np.mean([ind.objectives[0] for ind in population])),
                    'max': float(np.max([ind.objectives[0] for ind in population])),
                    'min': float(np.min([ind.objectives[0] for ind in population]))
                },
                'F2': {
                    'mean': float(np.mean([ind.objectives[1] for ind in population])),
                    'max': float(np.max([ind.objectives[1] for ind in population])),
                    'min': float(np.min([ind.objectives[1] for ind in population]))
                },
                'F3': {
                    'mean': float(np.mean([ind.objectives[2] for ind in population])),
                    'max': float(np.max([ind.objectives[2] for ind in population])),
                    'min': float(np.min([ind.objectives[2] for ind in population]))
                }
            }
        }

        with open(gen_dir / 'status.json', 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)

        if pareto_front:
            pareto_data = []
            for ind in pareto_front:
                if ind.muta_data:
                    pareto_data.append({
                        'file_name': ind.muta_data['file_name'],
                        'objectives': {
                            'F1': ind.objectives[0],
                            'F2': ind.objectives[1],
                            'F3': ind.objectives[2]
                        }
                    })

            with open(self.pareto_fronts_dir / f'gen_{generation:03d}_pareto.json', 'w', encoding='utf-8') as f:
                json.dump(pareto_data, f, indent=2, ensure_ascii=False)

    def get_prediction(self, image):
        """
        Get model prediction
        """
        with self.eval_lock:
            self.layer_output_input_dict.clear()
            self.layer_dict_keys_cnt = 0
            self.output = None

            with torch.no_grad():
                result = inference_model(self.model, image)
                layer_output_dict = copy.deepcopy(self.layer_output_input_dict)
                pred_sem_seg = result.pred_sem_seg.data.cpu().numpy()
            return layer_output_dict, pred_sem_seg

    def build_ori_seed_prediction_cache(self,datalist, desc):
        # if os.path.exists(self.TOOL_LOG_FILE_PATH + "/ori_seed_prediction_cache-old.pth"):
        #     self.ori_seed_prediction_cache = torch.load(self.TOOL_LOG_FILE_PATH + "/ori_seed_prediction_cache-old.pth")
        #     return None
        for idx in tqdm(range(len(datalist)), desc=desc, leave=True):
            image = datalist[idx][1]
            layer_output_dict, pred_sem_seg = self.get_prediction(image)
            cached_layer_dict = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in layer_output_dict.items()
            }
            self.ori_seed_prediction_cache[idx] = (cached_layer_dict, pred_sem_seg.copy())

        torch.save(self.ori_seed_prediction_cache, self.TOOL_LOG_FILE_PATH + "/ori_seed_prediction_cache-old.pth")
        exit()

    def build_muta_seed_prediction_cache(self, datalist, desc):
        for idx in tqdm(range(len(datalist)), desc=desc,leave=True):
            seed_idx, image = datalist[idx]
            layer_output_dict, pred_sem_seg = self.get_prediction(image)
            cached_layer_dict = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in layer_output_dict.items()
            }
            self.muta_seed_prediction_cache[seed_idx] = (cached_layer_dict, pred_sem_seg.copy())

    def get_prediction_with_cache(self, seed_idx, seed_prediction_cache, return_cpu_copy=False):
        assert seed_idx in seed_prediction_cache, f"Seed idx {seed_idx} not in cache. {seed_prediction_cache.keys()}"

        cached_layer_dict, cached_pred = seed_prediction_cache[seed_idx]

        if return_cpu_copy:
            layer_output_dict = {
                k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                for k, v in cached_layer_dict.items()
            }
            return layer_output_dict, cached_pred.copy()
        else:
            layer_output_dict = {
                k: v.clone().to(self.device) if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                for k, v in cached_layer_dict.items()
            }
            return layer_output_dict, cached_pred.copy()


    def check_stopping_criteria(self, generation: int, start_time: float, tknp_threshold: float = 0.20,
                                sbc_threshold: float = 0.90, adc_threshold: float = 0.95) -> Tuple[bool, str]:
        if generation < 5:
            return False, "Continue optimization"

        # 1. Maximum runtime
        elapsed_time = time.time() - start_time
        elapsed_hours = elapsed_time / 3600
        if elapsed_hours >= self.max_runtime_hours:
            return True, f"Maximum runtime reached: {elapsed_hours:.2f}h / {self.max_runtime_hours}h"

        # 2. Coverage threshold met
        tknp_old = self.history['coverage_history']['tknp'][-2]
        tknp_new = self.history['coverage_history']['tknp'][-1]
        sbc = self.boundary_coverage.get_coverage()
        adc = self.activation_coverage.current

        if (
                tknp_new / tknp_old) >= self.mostest_config.tknp_threshold and sbc >= self.mostest_config.sbc_threshold and adc >= self.mostest_config.adc_threshold:
            return True, f"Coverage thresholds met: TKNP={tknp_new:.3f}, SBC={sbc:.3f}, ADC={adc:.3f}"

        # 3. Coverage stagnation
        if len(self.history['coverage_history']['tknp']) >= 10:
            recent_tknp = self.history['coverage_history']['tknp'][-10:]
            if max(recent_tknp) - min(recent_tknp) < 0.001:
                return True, f"Coverage stagnation detected"

        # 4. Maximum generations
        if generation >= self.optimizer.max_generations:
            return True, f"Maximum generations reached: {generation}"

        return False, "Continue optimization"

    def run(self, datalist, verbose: bool = True) -> Dict:
        if verbose:
            console.print("\n[bold cyan]MOSTest: Multi-Objective Semantic Testing[/bold cyan]")
            console.print("[dim]" + "─" * 80 + "[/dim]")
            console.print(f"  Seeds: [cyan]{len(datalist)}[/cyan]")
            console.print(f"  Max Runtime: [cyan]{self.max_runtime_hours}h[/cyan]")

        self.build_ori_seed_prediction_cache(datalist,desc="Build ori seed prediction cache")

        start_time = time.time()
        if verbose:
            console.print(f"  Start: [dim]{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}[/dim]")
            console.print()

        population = Chromosome.create_random_population(
            self.optimizer.population_size,
            transform_configs=self.transform_sequence,
            spatial_enabled=False,
            single_transform_init=True,
            num_images=len(datalist)
        )

        if verbose:
            console.print("[cyan]→[/cyan] Initializing population")
            console.print(f"  Initial: [cyan]{len(population)}[/cyan] individuals")

        population = self._evaluate_population(population, datalist, "  First generatio")

        if verbose:
            console.print(f"  [green]✓[/green] Evaluated: [cyan]{len(population)}[/cyan] valid individuals\n")

        # Save initial population (Generation 0)
        if verbose:
            console.print("[dim]  Saving initial population...[/dim]")

        # check code logic
        self.save_generation_data(
            generation=0,
            population=population,
            save_images=True
        )

        # 3. Iterative optimization
        for generation in range(1, self.optimizer.max_generations + 1):
            if verbose:
                console.print(f"\n[bold cyan]Generation {generation}/{self.optimizer.max_generations}[/bold cyan]")
                console.print("[dim]" + "─" * 80 + "[/dim]")

            # 3.1 Create offspring
            offspring = self.optimizer.create_offspring(population, generation)
            console.print(f"\n[cyan]→[/cyan] Creating offspring: [dim]{len(population)} → {len(offspring)}[/dim]")

            # 3.2 Evaluate offspring
            offspring = self._evaluate_population(offspring, datalist, f"  Evaluate offspring Gen{generation}")

            # 3.3 Environmental selection
            combined = population + offspring
            population = self.optimizer.environmental_selection(
                combined, self.optimizer.population_size
            )

            console.print(f"  [green]✓[/green] Environmental selection: [cyan]{len(population)}[/cyan] individuals")

            # 3.4 Update coverage
            self.update_coverage(population)

            # 3.5 Record history
            tknp = self.tknp_coverage.current
            sbc = self.boundary_coverage.get_coverage()
            adc = self.activation_coverage.current

            self.history['coverage_history']['tknp'].append(tknp)
            self.history['coverage_history']['boundary'].append(sbc)
            self.history['coverage_history']['activation'].append(adc)

            # Record Pareto front
            fronts = self.optimizer.fast_non_dominated_sort(population)
            pareto_front = fronts[0] if fronts else []
            self.history['pareto_fronts'].append([
                {'objectives': ind.objectives} for ind in pareto_front
            ])

            # Save generation data
            self.save_generation_data(
                generation=generation,
                population=population,
                pareto_front=pareto_front,
            )

            uuids = [ind.muta_data["uuid"] for ind in population]
            self.muta_seed_prediction_cache = {key: self.muta_seed_prediction_cache[key] for key in uuids if key in uuids}



            if verbose:
                elapsed_time = time.time() - start_time
                elapsed_hours = elapsed_time / 3600

                console.print(
                    f"  Coverage: TKNP=[cyan]{tknp:.4f}[/cyan] SBC=[cyan]{sbc:.4f}[/cyan] ADC=[cyan]{adc:.4f}[/cyan]")
                console.print(
                    f"  Pareto Front: [cyan]{len(pareto_front)}[/cyan] | Runtime: [dim]{elapsed_hours:.2f}h / {self.max_runtime_hours}h[/dim]")
                console.print(f"  [green]✓[/green] Saved generation data")

            # 3.6 Check stopping criteria
            should_stop, reason = self.check_stopping_criteria(generation, start_time)
            if should_stop:
                if verbose:
                    console.print(f"\n[yellow]⚠[/yellow] Stopping: [dim]{reason}[/dim]")
                break

        # 4. Extract final test set
        if verbose:
            console.print(f"\n[cyan]→[/cyan] Extracting final test set...")

        fronts = self.optimizer.fast_non_dominated_sort(population)
        final_test_set = fronts[0] if fronts else population[:20]

        # 5. Generate test samples
        test_samples = []
        for chromosome in tqdm(final_test_set, disable=not verbose, desc="Generating samples"):
            muta_data = chromosome.muta_data

            test_samples.append({
                'original_file_name': chromosome.ori_file_name,
                'mutated_image': muta_data["muta_image"],
                'mutated_mask': muta_data["muta_mask"],
                'objectives': chromosome.objectives,
            })

        # 6. Save results
        self.save_results(test_samples)

        if verbose:
            # Calculate total runtime
            total_runtime = time.time() - start_time
            total_hours = total_runtime / 3600
            total_minutes = (total_runtime % 3600) / 60

            console.print(f"\n[bold green]✓ Optimization Completed[/bold green]")
            console.print("[dim]" + "─" * 80 + "[/dim]")
            console.print(
                f"  Runtime: [cyan]{total_hours:.2f}h[/cyan] ([dim]{int(total_hours)}h {int(total_minutes)}m[/dim])")
            console.print(f"  Samples: [cyan]{len(test_samples)}[/cyan]")
            console.print(
                f"  TKNP Coverage: [cyan]{self.tknp_coverage.current:.4f}[/cyan] ([dim]{int(self.tknp_coverage.current)} patterns[/dim])")
            console.print(f"  SBC Coverage: [cyan]{self.boundary_coverage.get_coverage():.4f}[/cyan]")
            console.print(f"  ADC Coverage: [cyan]{self.activation_coverage.current:.4f}[/cyan]")

        # Calculate total runtime
        total_runtime = time.time() - start_time

        return {
            'test_samples': test_samples,
            'history': self.history,
            'final_coverage': {
                'tknp': self.tknp_coverage.current,
                'sbc': self.boundary_coverage.get_coverage(),
                'adc': self.activation_coverage.current
            },
            'runtime': {
                'total_seconds': total_runtime,
                'total_hours': total_runtime / 3600,
                'start_time': start_time,
                'end_time': time.time()
            }
        }

    def save_results(self, test_samples: List[Dict]):
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        console.print(f"  [green]✓[/green] Results saved: [dim]{self.output_dir}[/dim]\n")

    def cleanup(self):
        for handle in self.handles:
            try:
                handle.remove()
            except:
                pass
        self.handles.clear()

        self.layer_output_input_dict.clear()

        self.layer_dict_keys_cnt = 0
        self.save_output = None

        console.print("[green]✓ Resources cleaned up[/green]")

