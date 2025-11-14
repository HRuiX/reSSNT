import copy
import pandas as pd
import utility 
from rich.console import Console
from rich.text import Text
import torch

console = Console()

diversity_cols = ['class', 'class-ALL', 'pixel', 'pixel-ALL', 'entropy', 'entropy-Var', 'TCE', 'TIE', 'IS', 'FID', 'LPIPS', 'SSIM']
factual_cols =['aAcc', 'mIoU', 'mAcc']
categories = ["bottom", "top", "random"] 
    
def get_max_idx(val):
    ret = ["bottom","top","random"]
    res = []
    max_val = max(val)
    for i in range(len(val)):
        if val[i] == max_val:
            res.append(ret[i])
    if len(res) == 1:
        return res[0]
    return res

def get_min_val(val):
    ret = ["bottom","top","random"]
    res = []
    min_val = min(val)
    for i in range(len(val)):
        if val[i] == min_val:
            res.append(ret[i])
    if len(res) == 1:
        return res[0]
    return res

def process_row(row):
    a = row["cov_rate"].split(",")

    if "%" in a[0]:
        a = (eval(a[0][1:-1]), eval(a[1][:-1]), eval(a[2][:-2]))
    else:
        a = (eval(a[0][1:]), eval(a[1]), eval(a[2][:-1]))

    row["cov_rate"] = get_max_idx(a)
    for c in diversity_cols:
        row[c] = get_max_idx(eval(row[c]))
    for c in factual_cols:
        row[c] = get_min_val(eval(row[c]))
    return row

def process_values(row, columns):
    max_val = []
    for c in columns:
        val = eval(row[1][c]) if "[" in row[1][c] else row[1][c]
        max_val.extend(val) if isinstance(val, list) else max_val.append(val)
    return max_val
    

def count_categories(values, categories):
    counts = [0] * len(categories)
    for val in values:
        if val in categories:
            counts[categories.index(val)] += 1
    return counts

    
def type_3(T):
    res = []
    for row in T.iterrows():
        diversity_vals = process_values(row, diversity_cols)
        diversity_counts = count_categories(diversity_vals, categories)
        
        factual_vals = process_values(row, factual_cols)
        factual_counts = count_categories(factual_vals, categories)
        
        res.append([
            row[1]["cov"], row[1]["hyper"], row[1]["cov_rate"],
            *diversity_counts, *factual_counts
        ])
    
    columns = ["cov", "hyper", "cov_rate", 
               "div-bottom", "div-Top", "div-random",
               "fact-bottom", "fact-top", "fact-random"]
    return pd.DataFrame(res, columns=columns)



def analyse_file(model_name,prefix_path): 
    save_path = "/".join(prefix_path.split("/")[:-1]) + "/analyse"
    utility.build_path(save_path)
    
    T_cov = pd.read_csv(f"{prefix_path}/Select_{model_name}_cov.csv").drop_duplicates(keep='last')
    T_div = pd.read_csv(f"{prefix_path}/Select_{model_name}-diversity.csv").drop_duplicates(keep='last')
    T_iou = pd.read_csv(f"{prefix_path}/select-{model_name}-acc-iou_Summery.csv").drop_duplicates(keep='last')
    
    T_cov["cov_rate"] = T_cov["End"]
    T = copy.deepcopy(T_cov[["cov","hyper","cov_rate"]])
    assert (T['hyper'] == T_div['hyper']).all
    assert (T['hyper'] == T_iou['hyper']).all
    
    for name in diversity_cols:
        T[name] = T_div[name]
    
    for name in factual_cols:
        T[name] = T_iou[name]
    
    T.to_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_1.csv", index=False)
    
    T = T.apply(process_row, axis=1)
    T.to_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_2.csv", index=False)
    
    T= type_3(T)
    T.to_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_3.csv", index=False)
    
    console.print(Text("   → ", style="dim"), Text(f"The file is analyzed and stored in folder", style="black"), Text(f"The file is analyzed and stored in folder", style="black"))
            
            
def deal_With_dataFrame(T,coverages_setting):
    covs = {}
    for t in T.iterrows():
        name = t[1]['Name'].split('-')
        if name[0] in covs.keys():
            if name[1] in covs[name[0]].keys():
                covs[name[0]][name[1]].update({name[2]:{'mIoU':t[1]['mIoU'],'mAcc':t[1]['mAcc'],'aAcc':t[1]['aAcc']}})
            else:
                covs[name[0]].update({name[1]:{name[2]:{'mIoU':t[1]['mIoU'],'mAcc':t[1]['mAcc'],'aAcc':t[1]['aAcc']}}})
        else:
            covs[name[0]] = {name[1]:{name[2]:{'mIoU':t[1]['mIoU'],'mAcc':t[1]['mAcc'],'aAcc':t[1]['aAcc']}}}
    res = []

    for cov in coverages_setting.keys():
        for hyper in coverages_setting[cov]:
            if hyper is not None and hyper == int(hyper):
                hyper = int(hyper)
            hyper = str(hyper)
            # print(covs[cov])
            # print(covs[cov][hyper])
            mIoU=f"({covs[cov][hyper]['bottom']['mIoU']},{covs[cov][hyper]['top']['mIoU']},{covs[cov][hyper]['random']['mIoU']})"
            mAcc=f"({covs[cov][hyper]['bottom']['mAcc']},{covs[cov][hyper]['top']['mAcc']},{covs[cov][hyper]['random']['mAcc']})"
            aAcc=f"({covs[cov][hyper]['bottom']['aAcc']},{covs[cov][hyper]['top']['aAcc']},{covs[cov][hyper]['random']['aAcc']})"
            res.append([cov,hyper,aAcc,mIoU,mAcc])
    res = pd.DataFrame(res,columns=["Cov","hyper","aAcc","mIoU","mAcc"])
    return res


def analysis_iou(res_dicts,Top_Save_Path,model_name,coverages_setting,cov_select_type=None):
    if cov_select_type == None:
        path = f"{Top_Save_Path}/Select-acc-iou_Original.pth"
    elif cov_select_type == "add":
        path = f"{Top_Save_Path}/Select-add-acc-iou_Original.pth"

    if res_dicts == []:
        res_dicts = torch.load(path)
    else: 
        utility.build_path(f"{Top_Save_Path}")
        torch.save(res_dicts, path)
    
    all_res, res_detail = [], []
    for res_dict in res_dicts:
        try:
            for name, metrics in res_dict.items():
                aacc, miou, macc = metrics["aAcc"], metrics["mIoU"], metrics["mAcc"]
                all_res.append([name, aacc, miou, macc])

                cls, iou, acc = metrics["class_table_data"][0][1], metrics["class_table_data"][1][1], \
                    metrics["class_table_data"][2][1]
                res = []
                for c, i, a in zip(cls, iou, acc):
                    res.append([c, i, a])
                res = pd.DataFrame(res, columns=["Class", f"{name}-IoU", f"{name}-Acc"])
                res_detail.append(res)
        except Exception as e:
            print("*="*100)
            print(e)
            print(model_name)
            print("*="*100)

    if cov_select_type == None:
        all_res = deal_With_dataFrame(pd.DataFrame(all_res, columns=["Name", "aAcc", "mIoU", "mAcc"]),coverages_setting=coverages_setting)
    elif cov_select_type == "add":
        all_res = pd.DataFrame(all_res, columns=["Name", "aAcc", "mIoU", "mAcc"])

    res_detail = pd.concat(res_detail, axis=1)

    if cov_select_type == None:
        # all_res.to_csv(f"{Top_Save_Path}/select-{model_name}-acc-iou_Summery.csv",index=False,mode='a')
        # res_detail.to_csv(f"{Top_Save_Path}/select-{model_name}-acc-iou_Detail.csv",index=False,mode='a')
        all_res.to_csv(f"{Top_Save_Path}/select-{model_name}-acc-iou_Summery.csv",index=False)
        res_detail.to_csv(f"{Top_Save_Path}/select-{model_name}-acc-iou_Detail.csv",index=False)
    elif cov_select_type == "add":
        all_res.to_csv(f"{Top_Save_Path}/Add-{model_name}-acc-iou_Summery.csv",index=False,mode='a')
        res_detail.to_csv(f"{Top_Save_Path}/Add-{model_name}-acc-iou_Detail.csv",index=False,mode='a')