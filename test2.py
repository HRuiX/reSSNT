import copy
import pandas as pd
import utility 
from rich.console import Console
from rich.text import Text

console = Console()

diversity_cols = ['class', 'class-ALL', 'pixel', 'pixel-ALL', 'entropy', 'entropy-Var', 'TCE', 'TIE', 'IS', 'FID', 'LPIPS']
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
    # a = row["cov_rate"].split(",")

    # if "%" in a[0]:
    #     a = (eval(a[0][1:-1]), eval(a[1][:-1]), eval(a[2][:-2]))
    # else:
    #     a = (eval(a[0][1:]), eval(a[1]), eval(a[2][:-1]))

    # row["cov_rate"] = get_max_idx(a)
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
    save_path = "/".join(prefix_path.split("/")[:-1]) + "/analyse0513"
    utility.build_path(save_path)
    
    # T_cov = pd.read_csv(f"{prefix_path}/Select_{model_name}_cov.csv").drop_duplicates(keep='last')
    T_div = pd.read_csv(f"{prefix_path}/Select_{model_name}-diversity.csv").drop_duplicates(keep='last')
    T_iou = pd.read_csv(f"{prefix_path}/select-{model_name}-acc-iou_Summery.csv").drop_duplicates(keep='last')
    T_iou['cov'] = T_iou['Cov']
    # T_cov["cov_rate"] = T_cov["End"]
    T = copy.deepcopy(T_div[["cov"]])
    assert (T['cov'] == T_div['cov']).all
    assert (T['cov'] == T_iou['cov']).all
    
    for name in diversity_cols:
        T[name] = T_div[name]
    
    for name in factual_cols:
        T[name] = T_iou[name]
    
    T.to_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_1.csv", index=False)
    
    T = T.apply(process_row, axis=1)
    T.to_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_2.csv", index=False)
    
    # T= type_3(T)
    # T.to_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_3.csv", index=False)
    
    console.print(Text("   → ", style="dim"), Text(f"The file is analyzed and stored in folder", style="black"), Text(f"The file is analyzed and stored in folder", style="black"))
    
    


data = {
    "ade20k":{
        "CNN": [
            "DeepLabV3Plus-R50-ade20k",
            "PSPNET_R101-ade20k",
            "FCN-HR48-ade20k"
        ],
        "Transformer": [
            "Mask2Former-Swin_S-ade20k",
            "Segmenter-Vit_t-ade20k",
            "Upernet_Deit_s16-ade20k"
        ],
        "Other": [
            "SegFormer-Mit_b0-ade20k",
            "Segnext_Mscan-b_1-ade20k",
            "Upernet_Convnext_base-ade20k",
        ]
    },
    "cityscapes":{
        "CNN": [
            "DeepLabV3Plus-R50-cityscapes",
            "PSPNET_R101-cityscapes",
            "FCN-HR48-cityscapes",
        ],
        "Transformer": [
            "Mask2Former-Swin_S-cityscapes",
            "Segmenter-Vit_t-cityscapes",
            "Vit-Deit_s-cityscapes"
        ],
        "Other": [
            "Upernet_Convnext_base-cityscapes",
            "SegFormer-Mit_b0-cityscapes",
            "Segnext_Mscan-b_1-cityscapes",
        ]
    }
}


# for dataset, values in data.items():
#     for model, model_names in values.items():
#         for model_name in model_names:
#             prefix_path = f"/home/ictt/xhr/code/DNNTesting/SSNT/output-div/{dataset}/{model}/{model_name}"
#             analyse_file(model_name=model_name, prefix_path=prefix_path)      
            


          
            
import pandas as pd
all_div_data = []
all_data_data = []
for dataset, values in data.items():
    for model,model_names in values.items():
        for model_name in model_names:
            prefix_path = f"/home/ictt/xhr/code/DNNTesting/SSNT/output-div/{dataset}/{model}/{model_name}"
            save_path = "/".join(prefix_path.split("/")[:-1]) + "/analyse0513"
            try:
                type1 = pd.read_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_1.csv")
                type2 = pd.read_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_2.csv")
                # type3 = pd.read_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_3.csv")
                
                # data = pd.concat([type1,type2,type3], axis=0)
                data = pd.concat([type1,type2], axis=0)
                data["model_name"] = [model_name]*data.shape[0]
                data["dataset"] = [dataset]*data.shape[0]
                data["model_type"] = [model]*data.shape[0]
                all_div_data.append(data)
                console.print(Text("   → ", style="dim"), Text(f"{model_name} OK", style="black"))
            except Exception as e:
                print(f"Error reading {model_name}: {e}")
                continue
            
            prefix_path = f"/home/ictt/xhr/code/DNNTesting/SSNT/output-data/{dataset}/{model}/{model_name}"
            save_path = "/".join(prefix_path.split("/")[:-1]) + "/analyse"
            try:
                type1 = pd.read_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_1.csv")
                type2 = pd.read_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_2.csv")
                type3 = pd.read_csv(f"{save_path}/Select_{model_name}_ALL_info_Type_3.csv")
                
                data = pd.concat([type1,type2,type3], axis=0)
                data["model_name"] = [model_name]*data.shape[0]
                data["dataset"] = [dataset]*data.shape[0]
                data["model_type"] = [model]*data.shape[0]
                all_data_data.append(data)
                console.print(Text("   → ", style="dim"), Text(f"{model_name} OK", style="black"))
            except Exception as e:
                print(f"Error reading {model_name}: {e}")
                continue
            
    
data_div = pd.concat(all_div_data, axis=0)
data = pd.concat(all_data_data, axis=0)

data_div.to_csv("/home/ictt/xhr/code/DNNTesting/SSNT/output/All_div_data0514.csv", index=False)
data.to_csv("/home/ictt/xhr/code/DNNTesting/SSNT/output/All_data0514.csv", index=False)
