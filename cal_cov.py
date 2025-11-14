import torch
import utility
import numpy as np
import pandas as pd


def cal_select_cov(prefix_path,coverages_setting, model_name, mode):
    cov_path = f"{prefix_path}/select_cov"

    df = []
    for cov, hypers in coverages_setting.items():
        for hyper in hypers:
            maxd, mean_val, variance_val, maxa, mina, avga = {}, {}, {}, {}, {}, {}
            for type in ["bottom", "random", "top"]:
                path1 = cov_path + f"/{cov}-{hyper}-{type}/{cov}/{model_name}_{mode}_{cov}_{hyper}_everytime_current.pth"
                try:
                    current = torch.load(path1)
                except Exception as e:
                    print(e)
                maxd[type] = current[-1]


                try:
                    path2 = cov_path + f"/{cov}-{hyper}-{type}/{cov}/{model_name}_{mode}_{cov}_{hyper}_every_pic_cover_record.pth"
                    pic_cover_record = torch.load(path2)
                    val = list(pic_cover_record.values())
                    if isinstance(val[0], torch.Tensor):
                        val = [d.item() for d in val]
                    mean_val[type] = np.mean(val)
                    variance_val[type] = np.var(val)
                    maxa[type] = max(val)
                    mina[type] = min(val)
                    avga[type] = sum(val) / len(val)
                except Exception as e:
                    print(e)

            if cov in ["NC", "KMNC", "SNAC", "NBC", "TKNC"]:
                df.append([cov, hyper, f"({maxd['bottom']:.2%}, {maxd['top']:.2%}, {maxd['random']:.2%})",
                           f"({mean_val['bottom']:.2%}, {mean_val['top']:.2%}, {mean_val['random']:.2%})",
                           f"({variance_val['bottom']:.2%}, {variance_val['top']:.2%}, {variance_val['random']:.2%})",
                           f"({maxa['bottom']:.2%}, {maxa['top']:.2%}, {maxa['random']:.2%})",
                           f"({mina['bottom']:.2%}, {mina['top']:.2%}, {mina['random']:.2%})",
                           f"({avga['bottom']:.2%}, {avga['top']:.2%}, {avga['random']:.2%})"])
            else:
                df.append([cov, hyper, f"({maxd['bottom']}, {maxd['top']}, {maxd['random']})",
                           f"({mean_val['bottom']}, {mean_val['top']}, {mean_val['random']})",
                           f"({variance_val['bottom']}, {variance_val['top']}, {variance_val['random']})",
                           f"({maxa['bottom']}, {maxa['top']}, {maxa['random']})",
                           f"({mina['bottom']}, {mina['top']}, {mina['random']})",
                           f"({avga['bottom']}, {avga['top']}, {avga['random']})"])

    df = pd.DataFrame(df, columns=["cov", "hyper", "End", "Hope", "Var", "Maxa", "Mina", "Avga"])
    df.to_csv(f"{prefix_path}/Select_{model_name}_cov.csv")


def cal_ALL_cov(data_save_path_prefix, model_name,mode,coverages_setting):
    prefix_path = f"{data_save_path_prefix}/all_test_cov"
    utility.build_path(prefix_path)
    
    df = []
    for cov, hypers in coverages_setting.items():
        for hyper in hypers:
            path1 = prefix_path + f"/{cov}/{model_name}_{mode}_{cov}_{hyper}_everytime_current.pth"
            current = torch.load(path1)
            maxd = current[-1]
            df.append([cov, hyper, maxd])

    df = pd.DataFrame(df, columns=["Cov", "hyper", "End"])
    df.to_csv(f"{data_save_path_prefix}/ALL_{model_name}_cov.csv")
    return df
    
    
    
def cal_cls_select_cov(prefix_path,model_name, mode, cov_save_path, coverages_setting):
    # cov_save_path = "/".join(prefix_path.split("/")[:-1])
    # utility.build_path(cov_save_path)

    df = []
    for cls in ["class", "class_new", "pixel", "entropy", "fid", "is", "lpips", "tce", "tie"]:
        for cov, hypers in coverages_setting.items():
            for hyper in hypers:
                maxd, mean_val, variance_val, maxa, mina, avga = {}, {}, {}, {}, {}, {}
                for type in ["bottom", "random", "top"]:
                    # path1 = prefix_path + f"/{cov}-{hyper}-{type}/test/{cov}/{kwargs['model_name']}_{kwargs['mode']}_{cov}_{hyper}_everytime_current.pth"
                    path1 = prefix_path + f"/{cls}-{type}/{cov}/{model_name}_{mode}_{cov}_{hyper}_everytime_current.pth"
                    
                    try:
                        current = torch.load(path1)
                    except Exception as e:
                        print(e)
                    maxd[type] = current[-1]

                    try:
                        path2 = prefix_path + f"/{cls}-{type}/{cov}/{model_name}_{mode}_{cov}_{hyper}_every_pic_cover_record.pth"
                        pic_cover_record = torch.load(path2)
                        val = list(pic_cover_record.values())
                        if isinstance(val[0], torch.Tensor):
                            val = [d.item() for d in val]
                        mean_val[type] = np.mean(val)
                        variance_val[type] = np.var(val)
                        maxa[type] = max(val)
                        mina[type] = min(val)
                        avga[type] = sum(val) / len(val)
                    except Exception as e:
                        print(e)

                if cov in ["NC", "KMNC", "SNAC", "NBC", "TKNC"]:
                    df.append([f"{cls}-{type}", cov, hyper, f"({maxd['bottom']:.2%}, {maxd['top']:.2%}, {maxd['random']:.2%})",
                            f"({mean_val['bottom']:.2%}, {mean_val['top']:.2%}, {mean_val['random']:.2%})",
                            f"({variance_val['bottom']:.2%}, {variance_val['top']:.2%}, {variance_val['random']:.2%})",
                            f"({maxa['bottom']:.2%}, {maxa['top']:.2%}, {maxa['random']:.2%})",
                            f"({mina['bottom']:.2%}, {mina['top']:.2%}, {mina['random']:.2%})",
                            f"({avga['bottom']:.2%}, {avga['top']:.2%}, {avga['random']:.2%})"])
                else:
                    df.append([f"{cls}-{type}", cov, hyper, f"({maxd['bottom']}, {maxd['top']}, {maxd['random']})",
                            f"({mean_val['bottom']}, {mean_val['top']}, {mean_val['random']})",
                            f"({variance_val['bottom']}, {variance_val['top']}, {variance_val['random']})",
                            f"({maxa['bottom']}, {maxa['top']}, {maxa['random']})",
                            f"({mina['bottom']}, {mina['top']}, {mina['random']})",
                            f"({avga['bottom']}, {avga['top']}, {avga['random']})"])

    df = pd.DataFrame(df, columns=["type", "cov", "hyper", "End", "Hope", "Var", "Maxa", "Mina", "Avga"])
    df.to_csv(f"{cov_save_path}/Select_{model_name}_cov.csv")