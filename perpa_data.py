import os
import torch
from tqdm.auto import tqdm
import pandas as pd
import utility
from rich.console import Console
from rich.text import Text
import random
import shutil

console = Console()


def get_file_name(label_path, image_path):
    label_files = utility.get_files(label_path)
    image_files = utility.get_files(image_path)

    return label_files, image_files


def matching_files_func(prefix, target_files):
    matching_files = []
    for target_file in target_files:
        if prefix in target_file:
            matching_files.append(target_file)
    return matching_files


def select_samples_files(prefix_path, coverages_setting, model_name, mode, N, top_flag="True"):
    file_save_path = prefix_path + "/select"
    os.makedirs(file_save_path, exist_ok=True)

    for cov, hypers in coverages_setting.items():
        for hyper in hypers:
            path2 = prefix_path + f"/{cov}/{model_name}_{mode}_{cov}_{hyper}_every_pic_cover_record.pth"
            data = torch.load(path2)
            df = pd.DataFrame(list(data.items()), columns=['Name', 'Value'])

            if top_flag:
                top = df.sort_values(by='Value', ascending=False).head(N)
                top_dict = top.set_index('Name')['Value'].to_dict()

            bottom = df.sort_values(by='Value', ascending=True).head(N)
            bottom_dict = bottom.set_index('Name')['Value'].to_dict()

            random = df.sample(n=N)
            random_dict = random.set_index('Name')['Value'].to_dict()

            torch.save(top_dict, file_save_path + f"/{cov}_{hyper}_top.pth")
            torch.save(bottom_dict, file_save_path + f"/{cov}_{hyper}_bottom.pth")
            torch.save(random_dict, file_save_path + f"/{cov}_{hyper}_random.pth")

    console.print(Text("   → ", style="dim"),
                  Text("The coverage value analysis of each file has been completed.", style="black"))


def build_data_path(dataset, dataset_path_prefix):
    if dataset == "cityscapes":
        label_path = f"{dataset_path_prefix}/gtFine/val/"
        image_path = f"{dataset_path_prefix}/leftImg8bit/val/"
    elif dataset == "ade20k":
        label_path = f"{dataset_path_prefix}/annotations/validation/"
        image_path = f"{dataset_path_prefix}/images/validation/"
    else:
        KeyError("dataset not supported!")

    return image_path, label_path


def move_to_path_samples(prefix_path, model_name, dataset, dataset_path_prefix, coverages_setting, top_flag="True"):
    prefix_path = f"{prefix_path}/select"

    progress = tqdm(total=len(coverages_setting), desc="   →  Moving files")
    res = []
    for cov, hypers in coverages_setting.items():
        for hyper in hypers:
            all = set()
            for type in ["bottom", "top", "random"]:

                if top_flag == "Fasle" and type == "top":
                    continue

                path2 = prefix_path + f"/{cov}_{hyper}_{type}.pth"
                data = torch.load(path2)
                if all == set():
                    all = set(data.keys())
                else:
                    all = all.intersection(set(data.keys()))

                val = data.values()
                res.append([f"{cov}-{hyper}-{type}", sum(val) / len(val)])

                save_root = f"{dataset_path_prefix}/diversity/{model_name}/select_files/{cov}-{hyper}/{type}/"
                prefix_image_path, prefix_label_path = build_data_path(dataset, save_root)
                for key, _ in data.items():
                    img_path, label_path = key[0].split("/")[-1], key[1].split("/")[-1]

                    new_img_path = f"{prefix_image_path}/{img_path}"
                    new_label_path = f"{prefix_label_path}/{label_path}"

                    copy_file_simple(key[0], new_img_path)
                    copy_file_simple(key[1], new_label_path)

            progress.update()
    res = pd.DataFrame(res, columns=["Type", "Value"])
    res.to_csv(f"{prefix_path}/Select_Avg_pic_cov.csv")
    console.print(Text("   → ", style="dim"), Text(f"File move completed.", style="black"))


def modify_config_file(model_name, dataset_path_prefix, config, coverages_setting, save_path=None,
                       modified_content=None):
    if save_path == None:
        save_path = f"{dataset_path_prefix}/diversity/{model_name}/config0515/"
    utility.build_path(save_path)

    with open(config, 'r') as file:
        content = file.read()

    for cov, hypers in coverages_setting.items():
        for hyper in hypers:
            for type in ["bottom", "top", "random"]:
                modified_content = content.replace(dataset_path_prefix,
                                                   f"{dataset_path_prefix}/diversity/{model_name}/select_files/{cov}-{hyper}/{type}/")

                with open(f'{save_path}/{cov}-{hyper}-{type}.py', 'w') as file:
                    file.write(modified_content)

    console.print(Text("   → ", style="dim"),
                  Text(f"The coinfig file has been rewritten, and the storage location is {save_path}", style="black"))

    return save_path


def move_select_add_cover_files(model_name, N, files_path, dataset, dataset_path_prefix):

    if dataset == "ade20k":
        img_save_folder = os.path.join(dataset_path_prefix, "images", "validation")
        label_save_folder = os.path.join(dataset_path_prefix, "annotations", "validation")
    elif dataset == "cityscapes":
        img_save_folder = os.path.join(dataset_path_prefix, "leftImg8bit", "val")
        label_save_folder = os.path.join(dataset_path_prefix, "gtFine", "val")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    label_files, image_files = get_file_name(label_save_folder, img_save_folder)
    label_files, image_files = set(label_files), set(image_files)

    for key, paths in files_path.items():
        save_path = f"{dataset_path_prefix}/diversity/{model_name}/select_files/{key}/top/"
        prefix_image_path, prefix_label_path = build_data_path(dataset, save_path)
        for k, path in paths.items():
            existing_paths_set = set(path)

            key_img_files, key_label_files = set(), set()
            print("existing_paths_set:", len(existing_paths_set))
            for a, b in existing_paths_set:
                key_img_files.add(a)
                key_label_files.add(b)

            while len(existing_paths_set) < N:
                tmp_label_files = label_files - key_label_files
                tmp_img_files = image_files - key_img_files

                img = random.choice(list(tmp_img_files))  # 随机选择
                tmp_img_files.remove(img)  # 从原集合中移除
                if dataset == "ade20k":
                    label = img.replace("/images/validation", "/annotations/validation").replace(".jpg", ".png")
                elif dataset == "cityscapes":
                    label = img.replace("/leftImg8bit", "/gtFine").replace("_leftImg8bit.png",
                                                                           "_gtFine_labelTrainIds.png")

                existing_paths_set.add((img, label))

            for img_path, label_path in path:
                old_img_path, old_label_path = img_path.split("/")[-1], label_path.split("/")[-1]

                new_img_path = f"{prefix_image_path}/{old_img_path}"
                new_label_path = f"{prefix_label_path}/{old_label_path}"

                copy_file_simple(img_path, new_img_path)
                copy_file_simple(label_path, new_label_path)

            print(f"    -> Processed key '{key}': {len(path)} file pairs copied to {save_path}")


def copy_file_simple(source_path, destination_path):
    if "/home/ictt/Documents/xhr" in source_path:
        source_path = source_path.replace("/home/ictt/Documents/xhr", "/home/ictt/Documents/xhr")
    if "/home/ictt/Documents/xhr" in destination_path:
        destination_path = destination_path.replace("/home/ictt/xhr", "/home/ictt/Documents/xhr")

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)


def modify_add_config_file(files_path, dataset_path_prefix, config, save_path=None, modified_content=None):
    with open(config, 'r') as file:
        content = file.read()

    for key, paths in files_path.items():
        save_path = f"{dataset_path_prefix}/add_cover_files/select_cov/{key}"
        modified_content = content.replace(dataset_path_prefix, save_path)

        utility.build_path(f"{dataset_path_prefix}/add_cover_files/config")
        with open(f'{dataset_path_prefix}/add_cover_files/config/{key}.py', 'w') as file:
            file.write(modified_content)

    console.print(Text("   → ", style="dim"),
                  Text(f"The coinfig file has been rewritten, and the storage location is {save_path}", style="black"))

    return f"{dataset_path_prefix}/add_cover_files/config"


def perpa_data_samples(prefix_path, model_name, config, mode, N, dataset, dataset_path_prefix, coverages_setting,
                       func=None, method="individual", files_path="", **kwargs):
    if method == "individual":
        if func:
            func(prefix_path, coverages_setting, model_name, mode, N)
        else:
            select_samples_files(prefix_path, coverages_setting, model_name, mode, N)
        move_to_path_samples(prefix_path, model_name, dataset, dataset_path_prefix, coverages_setting)
    elif method == "add":
        move_select_add_cover_files(model_name, N, files_path, dataset, dataset_path_prefix)
        select_samples_files(prefix_path, coverages_setting, model_name, mode, N, top_flag="Fasle")
        move_to_path_samples(prefix_path, model_name, dataset, dataset_path_prefix, coverages_setting, top_flag="Fasle")

    return modify_config_file(model_name, dataset_path_prefix, config, coverages_setting)

