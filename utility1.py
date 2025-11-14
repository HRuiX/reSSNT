import os
import json
import torch.nn as nn
import random
import numpy as np
import torchvision
import torch
import shutil
from rich.table import Table
from rich.console import Console
import os


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed leftImg8bit in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        # std_inv = 1 / (std + 1e-7)
        std_inv = 1 / std
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
        # return super().__call__(tensor)


def get_file_device(dataset, model_type):
    assert model_type in ["CNN", "Transformer", "Other"]
    res = []
    if model_type == "CNN":
        if dataset == "ade20k":
            res = [
                (f"DeepLabV3Plus-R50-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/ade20k/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/ade20k/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth"),
                 (f"PSPNET_R101-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/ade20k/resnest_s101-d8_pspnet_4xb4-160k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/ade20k/pspnet_s101-d8_512x512_160k_ade20k_20200807_145416-a6daa92a.pth"),
                (f"FCN-HR48-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/ade20k/fcn_hr48_4xb4-80k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/ade20k/fcn_hr48_512x512_80k_ade20k_20200614_193946-7ba5258d.pth"),
            ]
        elif dataset == "cityscapes":
            res = [
                (f"DeepLabV3Plus-R50-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/cityscapes/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth"),
                (f"PSPNET_R101-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/cityscapes/resnest_s101-d8_pspnet_4xb2-80k_cityscapes512x1024.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/cityscapes/pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth"),
                (f"FCN-HR48-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/cityscapes/fcn_hr18_4xb2-40k_cityscapes-512x1024.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/CNN/cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth"),
            ]
    elif model_type == "Transformer":
        if dataset == "ade20k":
            res = [
                (f"Mask2Former-Swin_S-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/ade20k/mask2former_swin-s_8xb2-160k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/ade20k/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth"),
                (f"Segmenter-Vit_t-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/ade20k/segmenter_vit-t_mask_8xb1-160k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/ade20k/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth"),
                (f"Upernet_Deit_s16-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/ade20k/vit_deit-s16_upernet_8xb2-80k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/ade20k/vit_deit-s16_upernet_8xb2-80k_ade20k-512x512.pth"),
            ]
        elif dataset == "cityscapes":
            res = [
                (f"Mask2Former-Swin_S-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024_20221127_143802-9ab177f6.pth"),
                (f"Segmenter-Vit_t-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/segmenter-vit-t-mask-cityspaces.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/segmenter-vit-t-mask-cityspaces.pth"),
                (f"Upernet-Deit_s16-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/vit_deit-s16_upernet-80k_cityscapes-512x1024.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/vit_deit-s16_upernet-cityspaces-8k.pth"),
            ]
    elif model_type == "Other":
        if dataset == "ade20k":
            res = [
                (f"SegFormer-Mit_b0-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/ade20k/segformer_mit-b0_8xb2-160k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth"),
                (f"Segnext_Mscan-b_1-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/ade20k/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/ade20k/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth"),
                (f"Upernet_Convnext_base-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/ade20k/upernet_convnext_base_ade20k-512x512.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth"),
            ]
        elif dataset == "cityscapes":
            res = [
                (f"Upernet_Convnext_base-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/cityscapes/upernet_convnext-base_cityscapes.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/cityscapes/upernet_convnext-base_cityscapes.pth"),
                (f"SegFormer-Mit_b0-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/cityscapes/segformer_mit-b0-cityspaces.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth"),
                (f"Segnext_Mscan-b_1-{dataset}",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/cityscapes/segnext_mscan-t-cityspaces.py",
                 "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/Other/cityscapes/segnext_mscan-t-cityspaces.pth"),
            ]

    if dataset == "cityscapes":
        dataset_path_prefix = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes/"
    elif dataset == "ade20k":
        dataset_path_prefix = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/ADEChallengeData2016/"

    return res, dataset_path_prefix


def dict_to_table(data_dict, title="IOU and Acc"):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Name", style="dim", width=20)
    table.add_column("Metric", style="green")

    for key, value in data_dict.items():
        table.add_row(str(key), str(value))

    return table


def seed_everything(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def build_path(path):
    if not isinstance(path, list):
        path = [path]
    for pa in path:
        if not os.path.exists(pa):
            os.makedirs(pa)


def copy_file(save_folder, image_matching_files):
    build_path(save_folder)
    for file_path in image_matching_files:
        shutil.copy(file_path, save_folder)


def get_files(folder_path, recursive=True):
    files = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            files.append(item_path)
        elif recursive and os.path.isdir(item_path):
            files.extend(get_files(item_path, recursive))
    return files


def write_to_txt(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


def write_to_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(str(data), file)


def read_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def read_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def is_valid(module):
    return (isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv3d)
            or isinstance(module, nn.GroupNorm)
            or isinstance(module, nn.Embedding)
            )


def iterate_module(name, module, name_list, module_list, name_label):
    """
    递归遍历模块结构，收集有效模块的名称和模块对象。

    该函数的主要职责是遍历每个模块（包括子模块），判断它们是否有效。
    如果模块有效，则将其名称和模块对象添加到相应的列表中。如果模块无效，
    则递归遍历其子模块，直到所有子模块都被处理或被判定为有效。

    参数：
    - name: 当前正在处理的模块名称。
    - module: 当前正在处理的模块对象。
    - name_list: 用于收集有效模块名称的列表。
    - module_list: 用于收集有效模块对象的列表。

    返回值：
    - name_list: 更新后的有效模块名称列表。
    - module_list: 更新后的有效模块对象列表。
    """
    if name_label != "":
        name = name_label + "-" + name

    # 检查当前模块是否有效
    if is_valid(module):
        # 如果有效，将模块名称和模块对象添加到列表中
        return name_list + [name], module_list + [module]
    else:
        # 如果无效，检查模块是否有子模块
        if len(list(module.named_children())):
            # 如果有子模块，递归调用此函数处理每个子模块
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module, name_list, module_list, name)
        return name_list, module_list


def get_model_layers(TOOL_LOG_FILE_PATH,model_name,model):
    """
    遍历模型的所有层，并创建一个包含模型所有层的字典。
    """
    path = f"{TOOL_LOG_FILE_PATH}/{model_name}_Layer_dictionary.pth"

    name_counter = {}
    layer_dict = {}

    cnt = 0
    # 遍历模型的每个子模块
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [], "")
        # 确保名称列表和模块列表长度一致
        assert len(name_list) == len(module_list)

        for i, _ in enumerate(name_list):
            module = module_list[i]
            # 获取模块的类名
            class_name = module.__class__.__name__

            # 如果类名不在名称计数器中，则初始化计数器
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                # 否则，增加计数器的值
                name_counter[class_name] += 1

            # 将模块添加到层字典中，键为类名和计数器组成的字符串
            layer_dict['%d-%d-%s-%s' % (cnt, name_counter[class_name], name_list[i], class_name)] = module
            cnt += 1

    torch.save(layer_dict, path)

    return layer_dict


def display_dict_as_columns(data, olddata, title="Data Overview"):
    """Display dictionary key-value pairs with optional comparison

    Args:
        data (dict): Dictionary to display
        title (str): Display title
        olddata (dict, optional): Previous data to compare against for changes
    """
    from rich.console import Console

    console = Console()

    # Print title
    console.print(f"\n[bold blue]{title}[/bold blue]")
    console.print("=" * len(title))

    # Display current data
    for key, value in data.items():
        # Format value (keep original format, no additional % sign)
        formatted_value = f"{float(value):.2f}"

        if olddata is not None:
            if key not in olddata:
                # New key - mark as added
                console.print(f"[bold green]+ {key}: {formatted_value}[/bold green]")
            elif olddata[key] != value:
                # Changed value - show change
                old_formatted = f"{float(olddata[key]):.2f}"
                change = float(value) - float(olddata[key])
                change_symbol = "↑" if change > 0 else "↓"
                change_formatted = f"{abs(change):.2f}"
                console.print(
                    f"[bold yellow]{key}: {formatted_value}[/bold yellow] [dim]({old_formatted} {change_symbol}{change_formatted})[/dim]")
            else:
                # No change
                console.print(f"{key}: {formatted_value}")
        else:
            # No comparison data
            console.print(f"{key}: {formatted_value}")

    # Show removed keys if olddata exists
    if olddata is not None:
        removed_keys = set(olddata.keys()) - set(data.keys())
        if removed_keys:
            console.print(f"\n[bold red]Removed:[/bold red]")
            for key in removed_keys:
                old_formatted = f"{float(olddata[key]):.2f}"
                console.print(f"[bold red]- {key}: {old_formatted}[/bold red]")