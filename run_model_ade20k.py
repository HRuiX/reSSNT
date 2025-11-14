import pandas as pd
import torch
import torch.nn as nn
from mmengine.runner import Runner
from mmengine.config import Config, DictAction
import os.path as osp
import logging
import warnings
from hook import SaveLayerInputOutputHook
warnings.filterwarnings("ignore", category=UserWarning)
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
logging.disable(logging.CRITICAL)
import os
from tqdm.auto import tqdm

import  multiprocessing
from multiprocessing import Pool, cpu_count

def get_files(target_folder):
    # 获取所有文件路径
    target_files = []
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            target_files.append(os.path.join(root, file))
    target_files = list(set(target_files))
    return target_files


def build_runner(model_config_file):
    cfg = Config.fromfile(model_config_file)

    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(model_config_file))[0])
    cfg.launcher = constants.LAUNCHER
    cfg.load_from = constants.MODEL_CHECKPOINT_FILE
    cfg.test_evaluator['output_dir'] = "./work_dirs/"
    cfg.test_evaluator['keep_results'] = True
    runner = Runner.from_cfg(cfg)
    runner.model.cfg = cfg
    return runner

def analyze_hooke(model_config_file):
    runner = build_runner(model_config_file=model_config_file)
    metrics = runner.val()

    aacc, miou, macc = metrics["aAcc"], metrics["mIoU"], metrics["mAcc"]
    cls, iou, acc = metrics["class_table_data"][0][1], metrics["class_table_data"][1][1], metrics["class_table_data"][2][1]
    res = []
    for c,i,a in zip(cls,iou,acc):
        res.append([c,i,a])
    res.append([aacc, miou, macc])

    name = model_config_file.split("/")[-1].split(".")[0]
    res = pd.DataFrame(res, columns=["Class", f"{name}-IoU", f"{name}-Acc"])

    return res

def analyze_hooke_main(model_name, type, model_config_file, save_path,cov,idx=0):
    runner = build_runner(model_config_file=model_config_file)
    # 创建自定义 Hook 实例
    save_layer_io_hook = SaveLayerInputOutputHook(runner.model, model_name, model_config_file, type, save_path=save_path,idx=idx,coverages=cov)
    save_layer_io_hook.run_for_size(runner.model)
    # 进行 output 的记录
    runner.register_hook(save_layer_io_hook)
    metrics = runner.val()

    aacc, miou, macc = metrics["aAcc"], metrics["mIoU"], metrics["mAcc"]
    cls, iou, acc =  metrics["class_table_data"][0][1], metrics["class_table_data"][1][1], \
    metrics["class_table_data"][2][1]
    res = []
    for c, i, a in zip(cls, iou, acc):
        res.append([c, i, a])
    res.append([aacc, miou, macc])

    name = model_config_file.split("/")[-1].split(".")[0]
    res = pd.DataFrame(res, columns=["Class", f"{name}-IoU", f"{name}-Acc"])

    return res

def process_file(file):
    return analyze_hooke(file)

def process_file_main(file,idx=0):
    file_name = file.split("/")[-1].replace(".py","")
    save_path = f"./output-data/ade20k/FCN-R50/choice_100_cov/{file_name}/"

    cov,hyper = file_name.split("-")[0],file_name.split("-")[1]
    if hyper != "None":
        hyper = float(hyper)

    cov = {cov:[hyper]}
    return analyze_hooke_main("FCN-R50-ade20k-0318", "test", file, save_path,idx=idx,cov=cov)

if __name__ == "__main__":
    # 获取所有文件
    # 设置多进程启动方式为 'spawn'
    multiprocessing.set_start_method('spawn', force=True)
    max_workers = 4

    # files = get_files("/home/ictt/xhr/code/DNNTesting/mmsegmentation/data/ADEChallengeData2016/diversity/FCN-R50/config/")

    # res = []
    # # 使用多进程并行处理文件
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(process_file_main, files[i], i) for i in range(len(files)) if "NBC" in files[i] or "NLC" in files[i]]
    #     for future in tqdm(as_completed(futures), total=len(files)):
    #         res.append(future.result())

    files = get_files("/home/ictt/xhr/code/DNNTesting/mmsegmentation/data/ADEChallengeData2016/diversity/FCN-R50/config/")
    res = []
    # 使用多进程并行处理文件
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file) for file in files]
        for future in tqdm(as_completed(futures), total=len(files)):
            res.append(future.result())
    # #
    torch.save(res,
               "/home/ictt/Documents/Rui/code/dnntesting/NSGen-ALL/output-log/ade20k/FCN-50R/ZanCun-Acc-FCN-R50-resluts-select-100-2.pth")
    res = pd.concat(res, axis=1)
    res.to_csv(
        "/home/ictt/Documents/Rui/code/dnntesting/NSGen-ALL/output-log/ade20k/FCN-50R/Acc-FCN-R50-resluts-select-100-2.csv")

    # file = "/home/ictt/xhr/code/DNNTesting/mmsegmentation/test_file/CNN/all_cov.py"
    # res = process_file_main(file)
    # torch.save(res, "/home/ictt/Documents/Rui/code/dnntesting/NSGen-ALL/output-log/ade20k/ZanCun-ALL-FCN-R50-ade20k.pth")
    # res.to_csv("/home/ictt/Documents/Rui/code/dnntesting/NSGen-ALL/output-log/ade20k/model-resluts-ALL-ade20k-FCN-R50.csv")


