import os
import json
import torch.nn as nn
import random
import numpy as np
import torchvision
import torch
import shutil
import cv2
import logging
from rich.table import Table
from rich.console import Console

# Configure logger
logger = logging.getLogger(__name__)


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
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


def get_file_device(dataset, model_type, ckpt_root="./ckpt", data_root="/home/ictt/xhr/code/DNNTesting/reSSNT/data"):
    """
    Get model configuration files and dataset paths based on dataset and model type.

    Args:
        dataset: Dataset name ("ade20k" or "cityscapes")
        model_type: Model architecture type ("CNN", "Transformer", or "Other")
        ckpt_root: Root directory for model checkpoints (default: "./ckpt")
        data_root: Root directory for datasets (default: "/home/ictt/xhr/code/DNNTesting/reSSNT/data")

    Returns:
        tuple: (list of model configs, dataset path prefix)
            - model configs: list of tuples (model_name, config_path, checkpoint_path)
            - dataset path prefix: full path to dataset directory
    """
    assert model_type in ["CNN", "Transformer", "Other"]
    res = []
    if model_type == "CNN":
        if dataset == "ade20k":
            res = [
                (
                    f"DeepLabV3Plus-R50-{dataset}",
                    f"{ckpt_root}/CNN/ade20k/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py",
                    f"{ckpt_root}/CNN/ade20k/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth",
                ),
                # (
                #     f"PSPNET_R101-{dataset}",
                #     f"{ckpt_root}/CNN/ade20k/resnest_s101-d8_pspnet_4xb4-160k_ade20k-512x512.py",
                #     f"{ckpt_root}/CNN/ade20k/pspnet_s101-d8_512x512_160k_ade20k_20200807_145416-a6daa92a.pth",
                # ),
                # (
                #     f"FCN-HR48-{dataset}",
                #     f"{ckpt_root}/CNN/ade20k/fcn_hr48_4xb4-80k_ade20k-512x512.py",
                #     f"{ckpt_root}/CNN/ade20k/fcn_hr48_512x512_80k_ade20k_20200614_193946-7ba5258d.pth",
                # ),
            ]
        elif dataset == "cityscapes":
            res = [
                (
                    f"DeepLabV3Plus-R50-{dataset}",
                    f"{ckpt_root}/CNN/cityscapes/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py",
                    f"{ckpt_root}/CNN/cityscapes/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth",
                ),
                # (
                #     f"PSPNET_R101-{dataset}",
                #     f"{ckpt_root}/CNN/cityscapes/resnest_s101-d8_pspnet_4xb2-80k_cityscapes512x1024.py",
                #     f"{ckpt_root}/CNN/cityscapes/pspnet_s101-d8_512x1024_80k_cityscapes_20200807_140631-c75f3b99.pth",
                # ),
                # (
                #     f"FCN-HR48-{dataset}",
                #     f"{ckpt_root}/CNN/cityscapes/fcn_hr18_4xb2-40k_cityscapes-512x1024.py",
                #     f"{ckpt_root}/CNN/cityscapes/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth",
                # ),
            ]
    elif model_type == "Transformer":
        if dataset == "ade20k":
            res = [
                (
                    f"Mask2Former-Swin_S-{dataset}",
                    f"{ckpt_root}/transformer/ade20k/mask2former_swin-s_8xb2-160k_ade20k-512x512.py",
                    f"{ckpt_root}/transformer/ade20k/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth",
                ),
                # (
                #     f"Segmenter-Vit_t-{dataset}",
                #     f"{ckpt_root}/transformer/ade20k/segmenter_vit-t_mask_8xb1-160k_ade20k-512x512.py",
                #     f"{ckpt_root}/transformer/ade20k/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth",
                # ),
                # (
                #     f"Upernet_Deit_s16-{dataset}",
                #     f"{ckpt_root}/transformer/ade20k/vit_deit-s16_upernet_8xb2-80k_ade20k-512x512.py",
                #     f"{ckpt_root}/transformer/ade20k/vit_deit-s16_upernet_8xb2-80k_ade20k-512x512.pth",
                # ),
            ]
        elif dataset == "cityscapes":
            res = [
                (
                    f"Mask2Former-Swin_S-{dataset}",
                    f"{ckpt_root}/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py",
                    f"{ckpt_root}/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024_20221127_143802-9ab177f6.pth",
                ),
                # (
                #     f"Segmenter-Vit_t-{dataset}",
                #     f"{ckpt_root}/transformer/cityscapes/segmenter-vit-t-mask-cityspaces.py",
                #     f"{ckpt_root}/transformer/cityscapes/segmenter-vit-t-mask-cityspaces.pth",
                # ),
                # (
                #     f"Upernet-Deit_s16-{dataset}",
                #     f"{ckpt_root}/transformer/cityscapes/vit_deit-s16_upernet-80k_cityscapes-512x1024.py",
                #     f"{ckpt_root}/transformer/cityscapes/vit_deit-s16_upernet-cityspaces-8k.pth",
                # ),
            ]
    elif model_type == "Other":
        if dataset == "ade20k":
            res = [
                (
                    f"SegFormer-Mit_b0-{dataset}",
                    f"{ckpt_root}/Other/ade20k/segformer_mit-b0_8xb2-160k_ade20k-512x512.py",
                    f"{ckpt_root}/Other/ade20k/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth",
                ),
                # (
                #     f"Segnext_Mscan-b_1-{dataset}",
                #     f"{ckpt_root}/Other/ade20k/segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512.py",
                #     f"{ckpt_root}/Other/ade20k/segnext_mscan-b_1x16_512x512_adamw_160k_ade20k_20230209_172053-b6f6c70c.pth",
                # ),
                # (
                #     f"Upernet_Convnext_base-{dataset}",
                #     f"{ckpt_root}/Other/ade20k/upernet_convnext_base_ade20k-512x512.py",
                #     f"{ckpt_root}/Other/ade20k/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth",
                # ),
            ]
        elif dataset == "cityscapes":
            res = [
                (
                    f"Upernet_Convnext_base-{dataset}",
                    f"{ckpt_root}/Other/cityscapes/upernet_convnext-base_cityscapes.py",
                    f"{ckpt_root}/Other/cityscapes/upernet_convnext-base_cityscapes.pth",
                ),
                # (
                #     f"SegFormer-Mit_b0-{dataset}",
                #     f"{ckpt_root}/Other/cityscapes/segformer_mit-b0-cityspaces.py",
                #     f"{ckpt_root}/Other/cityscapes/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth",
                # ),
                # (
                #     f"Segnext_Mscan-b_1-{dataset}",
                #     f"{ckpt_root}/Other/cityscapes/segnext_mscan-t-cityspaces.py",
                #     f"{ckpt_root}/Other/cityscapes/segnext_mscan-t-cityspaces.pth",
                # ),
            ]

    if dataset == "cityscapes":
        dataset_path_prefix = os.path.join(data_root, "cityscapes/")
    elif dataset == "ade20k":
        dataset_path_prefix = os.path.join(data_root, "ADEChallengeData2016/")

    return res, dataset_path_prefix


def dict_to_table(data_dict, title="IOU and Acc", use_rich=True):
    """
    Convert dictionary to a formatted table display.

    Args:
        data_dict: Dictionary to display
        title: Table title
        use_rich: If True, use Rich Table formatting; otherwise use plain text

    Returns:
        Rich Table object if use_rich=True, otherwise formatted string
    """
    if use_rich:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Name", style="dim", width=20)
        table.add_column("Metric", style="green")

        for key, value in data_dict.items():
            table.add_row(str(key), str(value))

        return table
    else:
        lines = [f"\n=== {title} ==="]
        for key, value in data_dict.items():
            lines.append(f"{key}: {value}")
        lines.append("=" * 50)
        return "\n".join(lines)


def display_dict_as_columns(data, olddata=None, title="Data Overview", use_rich=True):
    """
    Display dictionary key-value pairs with optional comparison.

    This enhanced display function shows:
    - New keys with + symbol
    - Changed values with change indicators (up/down arrows)
    - Unchanged values
    - Removed keys with - symbol

    Args:
        data (dict): Dictionary to display
        olddata (dict, optional): Previous data to compare against for changes
        title (str): Display title
        use_rich (bool): If True, use Rich console; otherwise use logging
    """
    if use_rich:
        console = Console()
        console.print(f"\n[bold blue]{title}[/bold blue]")
        console.print("=" * len(title))

        # Display current data
        for key, value in data.items():
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
    else:
        logger.info(f"\n{title}")
        logger.info("=" * len(title))

        # Display current data
        for key, value in data.items():
            # Format value (keep original format, no additional % sign)
            formatted_value = f"{float(value):.2f}"

            if olddata is not None:
                if key not in olddata:
                    # New key - mark as added
                    logger.info(f"+ {key}: {formatted_value}")
                elif olddata[key] != value:
                    # Changed value - show change
                    old_formatted = f"{float(olddata[key]):.2f}"
                    change = float(value) - float(olddata[key])
                    change_symbol = "↑" if change > 0 else "↓"
                    change_formatted = f"{abs(change):.2f}"
                    logger.info(f"{key}: {formatted_value} (was {old_formatted} {change_symbol}{change_formatted})")
                else:
                    # No change
                    logger.info(f"{key}: {formatted_value}")
            else:
                # No comparison data
                logger.info(f"{key}: {formatted_value}")

        # Show removed keys if olddata exists
        if olddata is not None:
            removed_keys = set(olddata.keys()) - set(data.keys())
            if removed_keys:
                logger.info("Removed:")
                for key in removed_keys:
                    old_formatted = f"{float(olddata[key]):.2f}"
                    logger.info(f"- {key}: {old_formatted}")


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (None to skip seeding)
    """
    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def build_path(path):
    """
    Create directory paths if they don't exist.

    Args:
        path: Single path string or list of path strings
    """
    if not isinstance(path, list):
        path = [path]
    for pa in path:
        if not os.path.exists(pa):
            os.makedirs(pa)


def copy_file(save_folder, image_matching_files):
    """
    Copy files to a destination folder.

    Args:
        save_folder: Destination folder path
        image_matching_files: List of file paths to copy
    """
    build_path(save_folder)
    for file_path in image_matching_files:
        shutil.copy(file_path, save_folder)


def get_files(folder_path, recursive=True):
    """
    Get all files in a folder.

    Args:
        folder_path: Path to folder
        recursive: If True, search recursively in subdirectories

    Returns:
        List of file paths
    """
    files = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            files.append(item_path)
        elif recursive and os.path.isdir(item_path):
            files.extend(get_files(item_path, recursive))
    return files


def write_to_txt(filename, content):
    """Write content to a text file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)


def write_to_json(filename, data):
    """Write data to a JSON file."""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(str(data), file)


def read_from_txt(filename):
    """Read content from a text file."""
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def read_from_json(filename):
    """Read data from a JSON file."""
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def is_valid(module):
    """
    Check if a module is a valid layer type for analysis.

    Args:
        module: PyTorch module to check

    Returns:
        True if module is a valid layer type, False otherwise
    """
    return (isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv3d)
            or isinstance(module, nn.GroupNorm)
            or isinstance(module, nn.Embedding)
            )


def iterate_module(name, module, name_list, module_list, name_label):
    """
    Recursively traverse module structure to collect valid module names and objects.

    This function traverses each module (including submodules), determines if they
    are valid. If a module is valid, its name and object are added to the respective
    lists. If a module is invalid, it recursively traverses its submodules until all
    submodules are processed or determined to be valid.

    Args:
        name: Name of the current module being processed
        module: Current module object being processed
        name_list: List to collect valid module names
        module_list: List to collect valid module objects
        name_label: Label prefix for hierarchical naming

    Returns:
        tuple: (name_list, module_list) with updated valid modules
    """
    if name_label != "":
        name = name_label + "-" + name

    if is_valid(module):
        return name_list + [name], module_list + [module]
    else:
        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list = iterate_module(
                    child_name, child_module, name_list, module_list, name
                )
        return name_list, module_list


def get_model_layers(model):
    """
    Traverse all layers of a model and create a dictionary containing all layers.

    Args:
        model: PyTorch model to analyze
        log_path: Path to save layer dictionary (optional)
        model_name: Name of model for file naming (required if log_path provided)

    Returns:
        Dictionary mapping layer identifiers to module objects
    """
    name_counter = {}
    layer_dict = {}

    cnt = 0
    # Traverse each submodule of the model
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [], "")
        # Ensure name list and module list have the same length
        assert len(name_list) == len(module_list)

        for i, _ in enumerate(name_list):
            module = module_list[i]
            # Get the class name of the module
            class_name = module.__class__.__name__

            # Initialize counter if class name not in counter
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                # Otherwise, increment the counter
                name_counter[class_name] += 1

            # Add module to layer dictionary with composite key
            layer_dict[
                "%d-%d-%s-%s"
                % (cnt, name_counter[class_name], name_list[i], class_name)
            ] = module
            cnt += 1

    return layer_dict


def fuzz_get_model_layers(model):
    """
    Traverse all layers of a model and create a dictionary containing all layers.

    This is a lightweight version for fuzzing that doesn't save to disk.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary mapping layer identifiers to module objects
    """
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

            layer_dict["%d-%d-%s-%s"% (cnt, name_counter[class_name], name_list[i], class_name)] = module
            cnt += 1
    return layer_dict


def build_img_path(top_save_path, path, old, epoch, ops, trans_cnt):
    """
    Build output image path for fuzzing results.

    Args:
        top_save_path: Top-level directory for saving
        path: Original image path
        old: Whether this is original (not transformed) image
        epoch: Epoch number
        ops: Operation name
        trans_cnt: Transformation count

    Returns:
        Constructed output path
    """
    path_idx = path.find("/data/")
    res_path = path.replace(path[: path_idx + len("/data")], top_save_path)

    check_path = os.path.dirname(res_path)
    if not os.path.exists(check_path):
        os.makedirs(check_path, exist_ok=True)

    if not old:
        path_parts = res_path.split("/")
        name_parts = path_parts[-1].split("_")
        name_parts.insert(1, f"epoch_{epoch}_{ops}_{trans_cnt}")
        path_parts[-1] = "_".join(name_parts)
        res_path = os.path.join(*path_parts)

    return res_path


def save_img(
    top_save_path, epoch, img, seg, img_path, seg_path, ops, trans_cnt, ori_pic=False
):
    """
    Save image and segmentation mask to disk.

    Args:
        top_save_path: Top-level directory for saving
        epoch: Epoch number
        img: Image array (RGB format)
        seg: Segmentation mask array
        img_path: Original image path (for naming)
        seg_path: Original segmentation path (for naming)
        ops: Operation name
        trans_cnt: Transformation count
        ori_pic: Whether this is original (not transformed) image

    Returns:
        True if save successful, False otherwise
    """
    img_path = build_img_path(top_save_path, img_path, ori_pic, epoch, ops, trans_cnt)
    seg_path = build_img_path(top_save_path, seg_path, ori_pic, epoch, ops, trans_cnt)

    success1 = cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    success2 = cv2.imwrite(seg_path, seg)

    if not (success1 and success2):
        print(f"File write failed: img={success1}, seg={success2}")
        return False
    return True