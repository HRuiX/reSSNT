# #!/usr/bin/env python3
# """
# 整理generations目录下所有gen_xxx文件夹的images和masks到统一的文件夹中 (优化版本)
# 主要优化: 多进程并行复制文件
# """
#
# import os
# import shutil
# from pathlib import Path
# from tqdm.auto import tqdm
# from multiprocessing import Pool, cpu_count
# from functools import partial
#
#
# def copy_file_task(args):
#     """单个文件复制任务"""
#     src_file, dest_file = args
#     try:
#         shutil.copy2(src_file, dest_file)
#         return True
#     except Exception as e:
#         print(f"复制失败 {src_file}: {e}")
#         return False
#
#
# def collect_files(gen_folders, base_type):
#     """
#     收集所有需要复制的文件
#     base_type: 'leftImg8bit' 或 'gtFine'
#     """
#     file_pairs = []
#
#     for gen_folder in gen_folders:
#         source_folder = gen_folder / "cityscapes" / base_type
#         if source_folder.exists() and source_folder.is_dir():
#             files = [f for f in source_folder.iterdir() if f.is_file()]
#             file_pairs.extend([(f, f.name) for f in files])
#
#     return file_pairs
#
#
# def organize_generations(base_path, output_path=None, num_workers=None):
#     """
#     将所有gen_xxx文件夹下的images和masks整理到统一的文件夹中 (多进程版本)
#
#     Args:
#         base_path: generations目录的路径
#         output_path: 输出目录路径，如果为None则在base_path同级创建organized文件夹
#         num_workers: 工作进程数，默认为CPU核心数
#     """
#     base_path = Path(base_path)
#
#     if not base_path.exists():
#         print(f"错误: 路径不存在: {base_path}")
#         return
#
#     # 设置工作进程数
#     if num_workers is None:
#         num_workers = cpu_count()
#
#     print(f"使用 {num_workers} 个进程进行并行复制")
#
#     # 设置输出路径
#     if output_path is None:
#         output_path = base_path.parent / "6-hours" / "cityscapes"
#     else:
#         output_path = Path(output_path)
#
#     # 创建输出目录
#     images_output = output_path / "leftImg8bit"
#     masks_output = output_path / "gtFine"
#
#     images_output.mkdir(parents=True, exist_ok=True)
#     masks_output.mkdir(parents=True, exist_ok=True)
#
#     print(f"源路径: {base_path}")
#     print(f"目标路径: {output_path}")
#     print(f"Images输出: {images_output}")
#     print(f"Masks输出: {masks_output}")
#     print("-" * 60)
#
#     # 查找所有gen_xxx文件夹
#     gen_folders = sorted([d for d in base_path.iterdir()
#                           if d.is_dir() and d.name.startswith("gen_")])
#     gen_folders = gen_folders[:len(gen_folders)//2]
#     if not gen_folders:
#         print("未找到任何gen_xxx文件夹")
#         return
#
#     print(f"找到 {len(gen_folders)} 个gen_xxx文件夹\n")
#
#     # 收集所有需要复制的文件
#     print("收集文件列表...")
#     image_pairs = collect_files(gen_folders, "leftImg8bit")
#     mask_pairs = collect_files(gen_folders, "gtFine")
#
#     print(f"找到 {len(image_pairs)} 个图像文件")
#     print(f"找到 {len(mask_pairs)} 个mask文件")
#     print()
#
#     # 准备复制任务 (源文件, 目标文件)
#     image_tasks = [(src, images_output / name) for src, name in image_pairs]
#     mask_tasks = [(src, masks_output / name) for src, name in mask_pairs]
#
#     # 使用多进程复制图像文件
#     print("复制图像文件...")
#     with Pool(num_workers) as pool:
#         results = list(tqdm(
#             pool.imap(copy_file_task, image_tasks),
#             total=len(image_tasks),
#             desc="Images"
#         ))
#     total_images = sum(results)
#
#     # 使用多进程复制mask文件
#     print("复制mask文件...")
#     with Pool(num_workers) as pool:
#         results = list(tqdm(
#             pool.imap(copy_file_task, mask_tasks),
#             total=len(mask_tasks),
#             desc="Masks"
#         ))
#     total_masks = sum(results)
#
#     # 输出统计信息
#     print("\n" + "=" * 60)
#     print("整理完成!")
#     print(f"处理的gen文件夹数: {len(gen_folders)}")
#     print(f"成功复制的图像: {total_images}/{len(image_tasks)}")
#     print(f"成功复制的masks: {total_masks}/{len(mask_tasks)}")
#     print(f"\n输出目录: {output_path}")
#
#
# if __name__ == "__main__":
#     # 默认路径
#     default_path = "/home/ictt/xhr/code/DNNTesting/reSSNT/mostest/mostest_output/output-data-1113-e767c360/cityscapes/Transformer/Mask2Former-Swin_S-cityscapes/generations/gen"
#
#     # 可以通过命令行参数指定路径
#     import sys
#
#     if len(sys.argv) > 1:
#         base_path = sys.argv[1]
#     else:
#         base_path = default_path
#
#     # 可以指定输出路径
#     output_path = None
#     if len(sys.argv) > 2:
#         output_path = sys.argv[2]
#
#     # 可以指定工作进程数
#     num_workers = 10
#     if len(sys.argv) > 3:
#         num_workers = int(sys.argv[3])
#
#     organize_generations(base_path, output_path, num_workers)


import os
import shutil
from pathlib import Path
import glob
from multiprocessing import Pool, cpu_count
from functools import partial


def process_single_gen_dir(gen_dir, dry_run=False):
    """
    处理单个 gen_* 目录

    Args:
        gen_dir: gen_* 目录路径
        dry_run: 是否只预览不实际执行

    Returns:
        处理结果信息
    """
    results = {
        'gen_dir': gen_dir,
        'leftimg_moved': 0,
        'gtfine_moved': 0,
        'errors': []
    }

    try:
        gen_name = os.path.basename(gen_dir)  # 例如: gen_1, gen_2, etc.

        # 处理 leftImg8bit
        leftimg_source = os.path.join(gen_dir, 'cityscapes/leftImg8bit')
        if os.path.exists(leftimg_source):
            # 创建目标目录
            leftimg_target = f'/home/ictt/xhr/code/DNNTesting/reSSNT/mostest/mostest_output/output-data-1113-e767c360/cityscapes/Transformer/Mask2Former-Swin_S-cityscapes/generations/gen/cityscapes/leftImg8bit/{gen_name}'
            if not dry_run:
                os.makedirs(leftimg_target, exist_ok=True)

            # 移动所有 PNG 文件
            png_files = glob.glob(os.path.join(leftimg_source, '*.png'))
            for png_file in png_files:
                filename = os.path.basename(png_file)
                target_path = os.path.join(leftimg_target, filename)

                if dry_run:
                    print(f"[预览] {png_file} -> {target_path}")
                else:
                    shutil.move(png_file, target_path)
                    print(f"[进程 {os.getpid()}] 移动: {filename} (leftImg8bit/{gen_name})")

                results['leftimg_moved'] += 1

        # 处理 gtFine
        gtfine_source = os.path.join(gen_dir, 'cityscapes/gtFine')
        if os.path.exists(gtfine_source):
            # 创建目标目录
            gtfine_target = f'/home/ictt/xhr/code/DNNTesting/reSSNT/mostest/mostest_output/output-data-1113-e767c360/cityscapes/Transformer/Mask2Former-Swin_S-cityscapes/generations/gen/cityscapes/gtFine/{gen_name}'
            if not dry_run:
                os.makedirs(gtfine_target, exist_ok=True)

            # 移动所有 PNG 文件
            png_files = glob.glob(os.path.join(gtfine_source, '*.png'))
            for png_file in png_files:
                filename = os.path.basename(png_file)
                target_path = os.path.join(gtfine_target, filename)

                if dry_run:
                    print(f"[预览] {png_file} -> {target_path}")
                else:
                    shutil.move(png_file, target_path)
                    print(f"[进程 {os.getpid()}] 移动: {filename} (gtFine/{gen_name})")

                results['gtfine_moved'] += 1

    except Exception as e:
        error_msg = f"处理 {gen_dir} 时出错: {str(e)}"
        results['errors'].append(error_msg)
        print(f"[错误] {error_msg}")

    return results


def cleanup_empty_dirs(gen_dirs):
    """
    清理空目录
    """
    print("\n清理空目录...")
    for gen_dir in gen_dirs:
        if os.path.exists(gen_dir):
            try:
                # 递归删除空目录
                for root, dirs, files in os.walk(gen_dir, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if os.path.exists(dir_path) and not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            print(f"删除空目录: {dir_path}")

                # 最后检查根目录
                if os.path.exists(gen_dir) and not os.listdir(gen_dir):
                    os.rmdir(gen_dir)
                    print(f"删除空目录: {gen_dir}")
            except Exception as e:
                print(f"清理目录时出错 {gen_dir}: {e}")


def reorganize_paths_multiprocess(num_processes=None, dry_run=False):
    """
    使用多进程重组文件路径结构

    Args:
        num_processes: 进程数，默认为CPU核心数
        dry_run: 是否只预览不实际执行
    """

    # 获取所有 gen_* 目录
    gen_dirs = sorted(glob.glob('/home/ictt/xhr/code/DNNTesting/reSSNT/mostest/mostest_output/output-data-1113-e767c360/cityscapes/Transformer/Mask2Former-Swin_S-cityscapes/generations/gen/gen_*'))

    if not gen_dirs:
        print("未找到任何 gen_* 目录")
        return

    print(f"找到 {len(gen_dirs)} 个 gen_* 目录")

    # 确定使用的进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(gen_dirs))

    print(f"使用 {num_processes} 个进程进行处理")
    print(f"模式: {'预览' if dry_run else '实际执行'}\n")

    # 创建进程池并处理
    with Pool(processes=num_processes) as pool:
        # 使用 partial 传递 dry_run 参数
        process_func = partial(process_single_gen_dir, dry_run=dry_run)

        # 并行处理所有目录
        results = pool.map(process_func, gen_dirs)

    # 汇总结果
    print("\n" + "=" * 60)
    print("处理完成！统计信息:")
    print("=" * 60)

    total_leftimg = sum(r['leftimg_moved'] for r in results)
    total_gtfine = sum(r['gtfine_moved'] for r in results)
    total_errors = sum(len(r['errors']) for r in results)

    print(f"leftImg8bit 文件数: {total_leftimg}")
    print(f"gtFine 文件数: {total_gtfine}")
    print(f"总文件数: {total_leftimg + total_gtfine}")

    if total_errors > 0:
        print(f"\n错误数: {total_errors}")
        print("错误详情:")
        for result in results:
            for error in result['errors']:
                print(f"  - {error}")

    # 清理空目录（仅在非预览模式下）
    if not dry_run:
        cleanup_empty_dirs(gen_dirs)

    print("\n路径重组完成!")


if __name__ == "__main__":
    print(f"当前工作目录: {os.getcwd()}")
    print(f"CPU核心数: {cpu_count()}")
    print()

    reorganize_paths_multiprocess(dry_run=False)
