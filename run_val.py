try:
    import torch
    import os
    from datetime import datetime
    import logging
    import warnings
    import multiprocessing
    import utility
    from config import CoverageTest

    # Configure logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    multiprocessing.set_start_method('spawn', force=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_printoptions(profile="default")

    checkpoint = "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024_20221127_143802-9ab177f6.pth"
    ori_config = "/home/ictt/xhr/code/DNNTesting/reSSNT/ckpt/transformer/cityscapes/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py"

    dataset = "cityscapes"
    model_type = "Transformer"
    model_name = f"Mask2Former-Swin_S-{dataset}"
    dataset_path_prefix = "/home/ictt/xhr/code/DNNTesting/reSSNT/data/cityscapes/"

    coverages_settings = {
        "NC": [0.75],
        "KMNC": [100],
        'SNAC': [None],
        'NBC': [None],
        'TKNC': [15],
        'CC': [19] if dataset == 'cityscapes' else [150],
        'TKNP': [25],
        'NLC': [None],
        "ADC": [100]
    }

    def print_test_header(test_name, test_type, config_path, index, total):
        """打印测试头部信息"""
        logger.info("=" * 80)
        logger.info(f"Test {index}/{total}")
        logger.info(f"Name: {test_name}")
        logger.info(f"Type: {test_type}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

    def print_test_complete(test_name):
        """打印测试完成信息"""
        logger.info(f"Completed: {test_name}")

    # 打印整体测试开始信息
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("Coverage Validation Test Suite")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # 计算总测试数量
    coverage_methods = ["CC-19","KMNC-100","NBC-None","SNAC-None","TKNC-15","TKNP-25","NLC-None"]
    test_types = ["ade_pic","cov_update"]
    total_tests = 3 + len(coverage_methods) * len(test_types)  # 3个基础测试 + 循环测试

    current_test = 0

    # Test 1: Original Coverage
    # current_test += 1
    # print_test_header("Original Coverage", "Baseline", ori_config, current_test, total_tests)
    # testTool = CoverageTest(model_name, dataset, model_type, ori_config, checkpoint,
    #                         dataset_path_prefix, save_data_path="1104-use-for-test/ori_cov",
    #                         coverages_setting=coverages_settings)
    # res_data = testTool.build_val_coverage_info()
    # torch.save(res_data, f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-data-1104-use-for-test/00results_save/Original_Coverage_info.pt")
    # print_test_complete("Original Coverage")

    # Test 3: MosTest Coverage (Pareto Front)
    current_test += 1
    mostest_config = "/home/ictt/xhr/code/DNNTesting/reSSNT/mostest/mostest_output/output-data-1114-7470e8a8/cityscapes/Transformer/Mask2Former-Swin_S-cityscapes/generations/mask2former_swin-s_8xb2-90k_cityscapes-512x1024-pareto.py"
    print_test_header("MosTest Coverage [Pareto Front]", "MosTest-PF", mostest_config, current_test, total_tests)
    testTool = CoverageTest(model_name, dataset, model_type, mostest_config, checkpoint,
                            dataset_path_prefix, save_data_path="1104-use-for-test/PF_mostest",
                            coverages_setting=coverages_settings)
    res_data = testTool.build_val_coverage_info()
    torch.save(res_data, f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-data-1104-use-for-test/00results_save/MosTest-PF_info.pt")
    print_test_complete("MosTest Coverage [Pareto Front]")

    # Test 2: MosTest Coverage (ALL)
    current_test += 1
    mostest_config = "/home/ictt/xhr/code/DNNTesting/reSSNT/mostest/mostest_output/output-data-1114-7470e8a8/cityscapes/Transformer/Mask2Former-Swin_S-cityscapes/generations/mask2former_swin-s_8xb2-90k_cityscapes-512x1024-6h.py"
    print_test_header("MosTest Coverage [ALL]", "MosTest-6h", mostest_config, current_test, total_tests)
    testTool = CoverageTest(model_name, dataset, model_type, mostest_config, checkpoint,
                            dataset_path_prefix, save_data_path="1104-use-for-test/ALL_mostest",
                            coverages_setting=coverages_settings)
    res_data = testTool.build_val_coverage_info()
    torch.save(res_data, f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-data-1104-use-for-test/00results_save/MosTest-All_info.pt")
    print_test_complete("MosTest Coverage [ALL]")


    # # Test 2: MosTest Coverage (ALL)
    # current_test += 1
    # mostest_config = "/home/ictt/xhr/code/DNNTesting/reSSNT/mostest/mostest_output/output-data-1114-7470e8a8/cityscapes/Transformer/Mask2Former-Swin_S-cityscapes/generations/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py"
    # print_test_header("MosTest Coverage [ALL]", "MosTest-All", mostest_config, current_test, total_tests)
    # testTool = CoverageTest(model_name, dataset, model_type, mostest_config, checkpoint,
    #                         dataset_path_prefix, save_data_path="1104-use-for-test/ALL_mostest",
    #                         coverages_setting=coverages_settings)
    # res_data = testTool.build_val_coverage_info()
    # torch.save(res_data, f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-data-1104-use-for-test/00results_save/MosTest-All_info.pt")


    # Test 4-N: Coverage Method Tests
    logger.info("=" * 80)
    logger.info("Coverage Method Specific Tests")
    logger.info("=" * 80)

    for k in coverage_methods:
        for ty in test_types:
            current_test += 1
            cov_config = f"/home/ictt/xhr/code/DNNTesting/reSSNT/fuzz-output-data-1104-use-new/{k}/cityscapes/Transformer/{ty}/{k}-muta-Mask2Former-Swin_S-cityscapes-config.py"

            test_name = f"{k} - {ty}"
            print_test_header(test_name, "Fuzz Test", cov_config, current_test, total_tests)

            testTool = CoverageTest(model_name, dataset, model_type, cov_config, checkpoint,
                                    dataset_path_prefix, save_data_path=f"1104-use-for-test/{k}-{ty}",
                                    coverages_setting=coverages_settings)
            res_data = testTool.build_val_coverage_info()
            torch.save(res_data, f"/home/ictt/xhr/code/DNNTesting/reSSNT/output-data-1104-use-for-test/00results_save/{test_name}_info.pt")
            print_test_complete(test_name)

    # 打印整体测试完成信息
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("=" * 80)
    logger.info("Test Suite Completed")
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {str(duration).split('.')[0]}")
    logger.info("=" * 80)
except Exception as e:
    print("\n" * 90)
    print(f"something went wrong: {e}")
    print("=" * 80)
    print("\n" * 90)






