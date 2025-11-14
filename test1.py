# # import pandas as pd
# # import torch
# # import os
# # from rich.text import Text
# # from rich.rule import Rule
# # from rich.console import Console
# # from mmengine.runner import Runner
# # from mmengine.config import Config
# # import os.path as osp
# # import logging
# # import warnings
# # from hook import SaveLayerInputOutputHook
# # from concurrent.futures import as_completed, ProcessPoolExecutor
# # import cal_cov
# # from tqdm.auto import tqdm
# # import multiprocessing
# # import utility
# # import perpa_data
# # import cal_diversity
# # from rich import print
# # import analyse
# # import copy
# # console = Console()
# # logging.disable(logging.CRITICAL)
# # multiprocessing.set_start_method('spawn', force=True)
# # warnings.filterwarnings("ignore", category=UserWarning)
# # torch.set_printoptions(profile="default")  # 避免张量输出过长
import json

# # def build_runner(model_config_file, model_check_file, mode):
# #     cfg = Config.fromfile(model_config_file)

# #     if mode == "train" or mode == "test_train":
# #         cfg.val_dataloader.dataset.data_prefix = cfg.train_dataloader.dataset.data_prefix
    
# #     cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(model_config_file))[0])
# #     cfg.launcher = "none"
# #     cfg.load_from = model_check_file
# #     cfg.test_evaluator['output_dir'] = "./work_dirs/"
# #     cfg.test_evaluator['keep_results'] = True
# #     runner = Runner.from_cfg(cfg)
# #     runner.model.cfg = cfg
# #     pipeline = []
# #     for pip in cfg.test_pipeline:
# #         if "LoadAnnotations" not in pip['type'] and "GenerateEdge" not in pip['type'] and "cat_max_ratio" not in pip.keys():
# #             pipeline.append(pip)
# #     return runner,pipeline


# # def analyze_hooke_main(model_config_file, model_check_file, save_path, idx=0, **kwargs):
# #     runner,pipeline = build_runner(model_config_file=model_config_file, model_check_file=model_check_file, mode=kwargs["mode"])
    
# #     if kwargs["coverages_setting"] != None:
# #         # 创建自定义 Hook 实例
# #         save_layer_io_hook = SaveLayerInputOutputHook(runner, model_config_file, pipeline=pipeline, save_path=save_path, idx=idx, **kwargs)
# #         # 进行 output 的记录
# #         runner.register_hook(save_layer_io_hook)
    
# #     metrics = runner.val()
# #     name = model_config_file.split("/")[-1].replace('.py',"")
    
# #     return {name: metrics}


# # def process_file_for_select(model_config_file, save_path, idx=0, use_cov=True,**kwargs):
# #     file_name = model_config_file.split("/")[-1].replace(".py", "")
# #     save_path = f"{save_path}/{file_name}/"
# #     if use_cov:
# #         cov, hyper = file_name.split("-")[0], file_name.split("-")[1]
# #         if hyper != "None":
# #             hyper = float(hyper)
# #         else:
# #             hyper = None
        
# #         if cov not in kwargs["coverages_setting"]:
# #             return None
# #         kwargs["coverages_setting"] = {cov: [hyper]}
# #     else:
# #         kwargs["coverages_setting"] = None
        
# #     return analyze_hooke_main(model_config_file=model_config_file, save_path=save_path, idx=idx, **kwargs)


# # def deal_With_dataFrame(T,**kwargs):
# #     covs = {}
# #     for t in T.iterrows():
# #         name = t[1]['Name'].split('-')
# #         if name[0] in covs.keys():
# #             if name[1] in covs[name[0]].keys():
# #                 covs[name[0]][name[1]].update({'mIoU':t[1]['mIoU'],'mAcc':t[1]['mAcc'],'aAcc':t[1]['aAcc']})
# #             else:
# #                 covs[name[0]].update({name[1]:{'mIoU':t[1]['mIoU'],'mAcc':t[1]['mAcc'],'aAcc':t[1]['aAcc']}})
# #         else:
# #             covs[name[0]] = {name[1]:{'mIoU':t[1]['mIoU'],'mAcc':t[1]['mAcc'],'aAcc':t[1]['aAcc']}}
# #     results = []
# #     for cov in ["class","pixel","entropy"]:
# #         mIoU=f"({covs[cov]['bottom']['mIoU']},{covs[cov]['top']['mIoU']},{covs[cov]['random']['mIoU']})"
# #         mAcc=f"({covs[cov]['bottom']['mAcc']},{covs[cov]['top']['mAcc']},{covs[cov]['random']['mAcc']})"
# #         aAcc=f"({covs[cov]['bottom']['aAcc']},{covs[cov]['top']['aAcc']},{covs[cov]['random']['aAcc']})"
# #         results.append([cov,aAcc,mIoU,mAcc])
# #     results = pd.DataFrame(results,columns=["Cov","aAcc","mIoU","mAcc"])
# #     return results


# # def analysis_iou(res_dicts,**kwargs):
# #     if res_dicts == []:
# #         res_dicts = torch.load(f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}/Select-acc-iou_Original.pth")
# #     else: 
# #         utility.build_path(f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}")
# #         torch.save(res_dicts, f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}/Select-acc-iou_Original.pth")
    
# #     all_res, res_detail = [], []
# #     for res_dict in res_dicts:
# #         for name, metrics in res_dict.items():
# #             aacc, miou, macc = metrics["aAcc"], metrics["mIoU"], metrics["mAcc"]
# #             all_res.append([name, aacc, miou, macc])

# #             cls, iou, acc = metrics["class_table_data"][0][1], metrics["class_table_data"][1][1], \
# #                 metrics["class_table_data"][2][1]
# #             res = []
# #             for c, i, a in zip(cls, iou, acc):
# #                 res.append([c, i, a])
# #             res = pd.DataFrame(res, columns=["Class", f"{name}-IoU", f"{name}-Acc"])
# #             res_detail.append(res)

# #     all_res = deal_With_dataFrame(pd.DataFrame(all_res, columns=["Name", "aAcc", "mIoU", "mAcc"]),**kwargs)
# #     res_detail = pd.concat(res_detail, axis=1)
    
# #     all_res.to_csv(f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}/select-{kwargs['model_name']}-acc-iou_Summery.csv")
# #     res_detail.to_csv(f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}/select-{kwargs['model_name']}-acc-iou_Detail.csv")


# # def modify_config_file(configs, **kwargs):
# #     types = ["top","bottom","random"]
# #     for i in range(len(types)):
# #         for div in ["class","pixel","entropy"]:
# #             # 读取原始文件
# #             with open(configs[1], 'r') as file:
# #                 content = file.read()
# #             utility.build_path(f"{kwargs['dataset_path_prefix']}/diversity/config/{configs[0]}")
# #             # 修改参数
# #             modified_content = content.replace(kwargs['dataset_path_prefix'], f"{kwargs['dataset_path_prefix']}/diversity/div/{div}/{types[i]}/")
# #             utility.build_path(f"{kwargs['dataset_path_prefix']}/diversity/div/config/{configs[0]}")
# #             # 保存修改后的内容到新文件
# #             with open(f"{kwargs['dataset_path_prefix']}/diversity/div/config/{configs[0]}/{div}-{types[i]}.py", 'w') as file:
# #                 file.write(modified_content)
 


# # if __name__ == "__main__":
# #     max_workers = 1
    
# #     DATASETS = [[],["ade20k"],["cityscapes"],["ade20k","cityscapes"]]
    
# #     for model_type in ["Other"]:
# #         for dataset in ["ade20k","cityscapes"]:
# #         # for dataset in ["cityscapes"]:
# #             model_infos,dataset_path_prefix = utility.get_file_device(dataset, model_type)
# #             for model_info in model_infos:
# #                 model_name,config,checkpoint = model_info[0],model_info[1],model_info[2]

# #                 coverages_setting = {
# #                     "NC": [0.25,0.5, 0.75],
# #                     "KMNC": [25,50,100],
# #                     'SNAC': [None],
# #                     'NBC': [None],
# #                     'TKNC': [5,10,15],
# #                     'TKNP': [10,25,50],
# #                     'CC': [5, 10, 15],
# #                     'NLC': [None],
# #                 }

# #                 N = 0
# #                 for cov, hypers in coverages_setting.items():
# #                     N += len(hypers)

# #                 kwargs = {
# #                     "model_name": model_name,
# #                     "dataset": dataset,
# #                     "device": "cuda:0",
# #                     "coverages_setting": coverages_setting,
# #                     "seed": 42,
# #                     "model_check_file": checkpoint,
# #                     "config": config,
# #                     "dataset_path_prefix": dataset_path_prefix,
# #                     "mode": "test",
# #                     "cover_n": N,
# #                     "choice_samples": 100,
# #                     "model_type":model_type,
# #                     "Top_Save_Path":f"./output-div/{dataset}/{model_type}"
# #                 }

# #                 utility.seed_everything(kwargs["seed"])

# #                 console.print(Text(f"\nWe're going to start testing model {model_name} now!", style="bold bright_cyan"))
# #                 console.print(Rule(style="dim cyan"))
                
              
# #                 console.print(Text("Step 1. Preliminary construction of coverage measurement information.", style="bold green"))
# #                 save_path = f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}/all_train_cov"
# #                 utility.build_path(save_path)
# #                 kwargs["mode"] = "train"
# #                 res = analyze_hooke_main(model_config_file=config, save_path=save_path, **kwargs)
# #                 print(utility.dict_to_table(res[config.split("/")[-1].replace('.py',"")]))
# #                 file_path = f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}"
# #                 utility.build_path(file_path)
                
# #                 modify_config_file(model_info, **kwargs)
                
# #                 # Step 4: 对3个进行覆盖度量的计算
# #                 console.print(Text(f"Step 4. Calculate coverage metrics for three {kwargs['choice_samples']}s.", style="bold green"))
# #                 file_path = f"{kwargs['dataset_path_prefix']}/diversity/div/config/{model_name}"
# #                 files = utility.get_files(file_path)
                
# #                 kwargs["mode"] = "test"
# #                 save_path = f"{kwargs['Top_Save_Path']}/{kwargs['model_name']}/select_cov"
# #                 utility.build_path(save_path)
# #                 kwargs["ALL_FILE_NUM"] = len(files)
                
# #                 res = []
# #                 for i in tqdm(range(len(files)), desc="   → Executing collection of coverage metrics for various datasets"):
# #                     file_name = files[i].split("/")[-1].replace(".py", "")
# #                     save_path_x = f"{save_path}/{file_name}/"
# #                     res.append(analyze_hooke_main(model_config_file=files[i], save_path=save_path_x, **kwargs))
            
# #                 res = analysis_iou(res,**kwargs)
# #                 save_path = f"./output-div/{dataset}/{model_type}/{kwargs['model_name']}/select_cov"
# #                 cal_cov.cal_cls_select_cov(save_path, **kwargs) 

# #                 # Step 5: 对3个进行多样性的计算
# #                 console.print(Text(f"Step 5. Calculate diversity for three {kwargs['choice_samples']}s", style="bold green"))
# #                 cal_diversity.cal_diversity_div(**kwargs)


# import os
# import re
# from pathlib import Path

# def replace_parameters_in_file(file_path):
#     """
#     替换单个文件中的batch_size和num_workers参数
#     """
#     try:
#         # 读取文件内容
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         # 保存原始内容用于比较
#         original_content = content
        
#         # 替换batch_size=任意数字为batch_size=1
#         # 匹配模式：batch_size=数字（可能包含空格）
#         content = re.sub(r'batch_size\s*=\s*\d+,', 'batch_size=1,', content)
        
#         # 替换num_workers=任意数字为num_workers=20
#         # 匹配模式：num_workers=数字（可能包含空格）
#         content = re.sub(r'num_workers\s*=\s*\d+,', 'num_workers=10,', content)
        
#         # 只有内容发生变化时才写入文件
#         if content != original_content:
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 f.write(content)
#             return True  # 表示文件被修改
#         return False  # 表示文件未被修改
        
#     except Exception as e:
#         print(f"处理文件 {file_path} 时出错: {e}")
#         return False

# def find_and_replace_in_directory(directory_path):
#     """
#     查找目录下所有.py文件并替换参数
#     """
#     directory = Path(directory_path)
    
#     if not directory.exists():
#         print(f"错误：目录 {directory_path} 不存在")
#         return
    
#     if not directory.is_dir():
#         print(f"错误：{directory_path} 不是一个目录")
#         return
    
#     # 统计信息
#     total_files = 0
#     modified_files = 0
    
#     print(f"开始处理目录: {directory_path}")
#     print("-" * 50)
    
#     # 递归查找所有.py文件
#     for py_file in directory.rglob('*.py'):
#         total_files += 1
#         print(f"处理文件: {py_file}")
        
#         if replace_parameters_in_file(py_file):
#             modified_files += 1
#             print(f"  ✓ 已修改")
#         else:
#             print(f"  - 无需修改")
    
#     print("-" * 50)
#     print(f"处理完成！")
#     print(f"总共处理了 {total_files} 个Python文件")
#     print(f"修改了 {modified_files} 个文件")

# def main():
#     """
#     主函数
#     """
#     print("Python文件参数批量替换工具")
#     print("功能：将所有.py文件中的batch_size=n替换为batch_size=1，num_workers=n替换为num_workers=20")
#     print("=" * 60)
    
#     # 获取用户输入的目录路径
#     while True:
#         directory_path = input("请输入要处理的目录路径（输入'.'表示当前目录）: ").strip()
        
#         if directory_path == "":
#             print("路径不能为空，请重新输入")
#             continue
        
#         # 转换为绝对路径
#         directory_path = os.path.abspath(directory_path)
        
#         # 确认操作
#         print(f"\n将要处理目录: {directory_path}")
#         confirm = input("确认要继续吗？(y/n): ").strip().lower()
        
#         if confirm in ['y', 'yes', '是', 'Y']:
#             break
#         elif confirm in ['n', 'no', '否', 'N']:
#             print("操作已取消")
#             return
#         else:
#             print("请输入 y 或 n")
#             continue
    
#     # 执行替换操作
#     find_and_replace_in_directory(directory_path)

# if __name__ == "__main__":
#     main()


import utility
import pandas as pd
import shutil
import os
import copy

DATASETS = [[], ["ade20k"], ["cityscapes"], ["ade20k", "cityscapes"]]
MODEL_TYPE = ["", "CNN", "Transformer", "Other"]
data = []

datasets_idx = 3
for model_type_idx in [1, 2, 3]:
    model_type = MODEL_TYPE[model_type_idx]
    for dataset in DATASETS[datasets_idx]:
        model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
        for model_info in model_infos:
            model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
            coverages_settings =  {
                    "NC": [0.75],
                    "KMNC": [100],
                    'SNAC': [None],
                    'NBC': [None],
                    'TKNC': [15],
                    'CC': [19] if dataset == 'cityscapes' else [150],
                    'TKNP': [25],
                    'NLC': [None],
                }
            
            for key,value in coverages_settings.items():
                    for v in value:
                        if "." in str(v): vs = str(v).replace(".", "")
                        else: vs = str(v)
                        
                        files = []

                        pathfix = "./output-fuzz-data-0917"


                        # # ==============================================
                        # # 统计每个模型的测试数据量 ori_data 及 muta_data
                        # # ==============================================
                        # ori_data = []
                        # muta_data = []
                        # from tqdm.auto import tqdm
                        # for epoch in tqdm(range(2000),desc=f"{key}-{vs}-{dataset}"):
                        #     for typ in ["ori","muta"]:
                        #         if dataset == 'cityscapes':
                        #             path = f"{pathfix}/{key}-{vs}/cityscapes/{model_type}/fuzz_test/epoch-{epoch}/{typ}/cityscapes/leftImg8bit/val"
                        #         else:
                        #             path = f"{pathfix}/{key}-{vs}/ade20k/{model_type}/fuzz_test/epoch-{epoch}/{typ}/ADEChallengeData2016/leftImg8bit/validation"
                        #
                        #         try:
                        #             files = utility.get_files(path)
                        #
                        #             for file in files:
                        #                 if typ == "ori":
                        #                     ori_data.append(
                        #                         [key, v, dataset, model_type, model_name, f"epoch-{epoch}", file])
                        #                 else:
                        #                     muta_data.append(
                        #                         [key, v, dataset, model_type, model_name, f"epoch-{epoch}", file])
                        #         except:
                        #             continue
                        #
                        # if dataset == 'cityscapes':
                        #     path = f"{pathfix}/{key}-{vs}/cityscapes/{model_type}/fuzz_test/"
                        # else:
                        #     path = f"{pathfix}/{key}-{vs}/ade20k/{model_type}/fuzz_test/"
                        #
                        # path_ori = f"{path}/ori_data.xlsx"
                        # path_muta = f"{path}/muta_data.xlsx"
                        # import torch
                        # torch.save(ori_data, path_ori)
                        # torch.save(muta_data, path_ori)
                        # ori_data = pd.DataFrame(ori_data)
                        # muta_data = pd.DataFrame(muta_data)
                        # ori_data.to_excel(path_ori, index=False)
                        # muta_data.to_excel(path_muta, index=False)
                        #
                        # data = {
                        #     "cov": key,
                        #     "hyper": v,
                        #     "dataset": dataset,
                        #     "model_type": model_type,
                        #     "model_name": model_name,
                        #     "epoch": epoch,
                        #     "ori": len(ori_data),
                        #     "muta": len(muta_data)
                        # }
                        #
                        # import json
                        # with open(f"{path}/summary.json", "w") as file:
                        #     json.dump(data, file)

                        # ==============================================
                        # 删除文件使用
                        # ==============================================
                        from tqdm.auto import tqdm
                        for epoch in tqdm(range(2000), desc=f"{key}-{vs}-{dataset}"):
                            if dataset == 'cityscapes':
                                path = f"{pathfix}/{key}-{vs}/cityscapes/{model_type}/fuzz_test/epoch-{epoch}/"
                            else:
                                path = f"{pathfix}/{key}-{vs}/ade20k/{model_type}/fuzz_test/epoch-{epoch}/"

                            try:
                                shutil.rmtree(path)
                            except:
                                continue

                            # ori = pd.read_excel(path_ori)
                            # muta = pd.read_excel(path_muta)
                            # cov = pd.read_excel(path_cov)
                            #
                            #     data.append([key, v, dataset, model_type, model_name, len(cov), len(ori), len(muta)])
                            #     print(key, v, dataset, model_type, model_name, len(cov), len(ori), len(muta))
                            #
                        # if dataset == 'cityscapes':
                        #     path = f"{pathfix}/{key}-{vs}/cityscapes/{model_type}/fuzz_test/"
                        # else:
                        #     path = f"{pathfix}/{key}-{vs}/ade20k/{model_type}/fuzz_test/"
                        
                        # try:
                        #     # save_path = f"{pathfix}/config/{model_type}-{dataset}-{key}-{vs}.py"
                        #     # if os.path.exists(path):
                        #     #     with open(config, 'r') as file:
                        #     #         content = file.read()
                        #     #         content = content.replace(dataset_path_prefix[:-1], path)
                        #     #         with open(save_path, 'w') as file:
                        #     #             file.write(content)
                        #     # else:
                        #     #     print(f"文件夹 '{path}' 不存在")
                        #
                        #     # 获取文件信息
                        #     # files = utility.get_files(path)
                        #     # for file in files:
                        #     #     data.append([key,v,dataset,model_type,model_name,f"epoch-{epoch}",file])
                        #
                        #     # 删除文件
                        #     # if os.path.exists(path):
                        #     #     shutil.rmtree(path)
                        #     #     print(f"文件夹 '{path}' 已成功删除")
                        #     # else:
                        #     #     print(f"文件夹 '{path}' 不存在")
                        #
                        #     path_ori = f"{path}/ori_data.xlsx"
                        #     path_muta = f"{path}/muta_data.xlsx"
                        #     path_cov = f"{path}/cov_update/{key}-{v}/data.xlsx"
                        #
                        #     ori = pd.read_excel(path_ori)
                        #     muta = pd.read_excel(path_muta)
                        #     cov = pd.read_excel(path_cov)
                        #
                        #     data.append([key, v, dataset, model_type,  model_name, len(cov), len(ori), len(muta)])
                        #     print(key, v, dataset, model_type,  model_name, len(cov), len(ori), len(muta))
                        #
                        #     del ori, muta, cov
                        #     import gc
                        #     gc.collect()
                        #
                        # except FileNotFoundError:
                        #     continue
                        
                        # print(model_type, dataset, key, v, len(files))
                        
                        # data = pd.DataFrame(data, columns=["cov","hyper",'dataset', 'model_type', 'model_name', "epoch", 'file'])
                        # data.to_excel(f"{pathfix}/{key}-{vs}/{dataset}/{model_type}/fuzz_test/muta_data.xlsx", index=False)
# import torch
# torch.save(data, f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-fuzz-data-0714/summary.pt")
# data = pd.DataFrame(data, columns=["cov","hyper",'dataset', 'model_type', 'model_name', "update", 'ori', "muta"])
# data.to_excel(f"{pathfix}/summary.xlsx", index=False)
                    