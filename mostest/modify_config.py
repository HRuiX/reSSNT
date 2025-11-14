def modify_config_file(config, save_path, modified_content):
    with open(config, 'r') as file:
        content = file.read()
        modified_content = content.replace(dataset_path_prefix,modified_content)
        with open(save_path, 'w') as file1:
            file1.write(modified_content)
    print(f"The coinfig file has been rewritten, and the storage location is {save_path}")
import utility
dataset = "cityscapes"
model_type = "Transformer"
model_infos, dataset_path_prefix = utility.get_file_device(dataset, model_type)
num_classes = 150 if dataset == "ade20k" else 19
coverages_settings = {
    "NC": 0.75,
    "KMNC": 100,
    'SNAC': None,
    'NBC': None,
    'TKNC': 15,
    'CC': 19 if dataset == 'cityscapes' else 150,
    'TKNP': 25,
    'NLC': None,
}

prefix_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/fuzz-output-data-1104-use-new"

for model_info in model_infos:
    model_name, config, checkpoint = model_info[0], model_info[1], model_info[2]
    for key, value in coverages_settings.items():
        modify_content = f"{prefix_path}/{key}-{value}/{dataset}/{model_type}/cov_update/muta/cityscapes"
        save_path = f"{prefix_path}/{key}-{value}/{dataset}/{model_type}/cov_update/{key}-{value}-muta-{model_name}-config.py"
        modify_config_file(config, save_path, modify_content)

        modify_content = f"{prefix_path}/{key}-{value}/{dataset}/{model_type}/ade_pic/cityscapes"
        save_path = f"{prefix_path}/{key}-{value}/{dataset}/{model_type}/ade_pic/{key}-{value}-muta-{model_name}-config.py"
        modify_config_file(config, save_path, modify_content)

