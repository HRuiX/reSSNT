import argparse
import torch
import os
from typing import List, Optional
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

multiprocessing.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_printoptions(profile="default")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Unified Coverage Testing Runner for Semantic Segmentation Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-types CNN Transformer Other --datasets ade20k cityscapes
  %(prog)s --model-types CNN --datasets cityscapes --cov-select-type add
  %(prog)s --model-types Transformer --datasets ade20k --mode diversity
  %(prog)s --model-types Other --datasets cityscapes --save-path custom-results
        """,
    )

    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["CNN", "Transformer", "Other"],
        default=["CNN", "Transformer", "Other"],
        help="Model types to test (default: all types)",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ade20k", "cityscapes"],
        default=["ade20k", "cityscapes"],
        help="Datasets to use (default: both datasets)",
    )

    parser.add_argument(
        "--mode",
        choices=["standard", "add", "diversity"],
        default="standard",
        help="Testing mode: standard, add, or diversity (default: standard)",
    )

    parser.add_argument(
        "--cov-select-type",
        type=str,
        default=None,
        help='Coverage selection type (e.g., "add" for additive coverage)',
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="0908-all",
        help="Data save path prefix (default: 0908-all)",
    )

    parser.add_argument(
        "--filter-model",
        type=str,
        default=None,
        help='Only run specific model (e.g., "Mask2Former-Swin_S-cityscapes")',
    )

    parser.add_argument(
        "--skip-steps",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=[],
        help="Steps to skip (1: pre-build, 2: val coverage, 3: select, 4: select coverage, 5: diversity, 6: analyze)",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of worker processes (default: 1)",
    )

    parser.add_argument(
        "--coverage-settings",
        type=str,
        default=None,
        help="Custom coverage settings as JSON string (optional)",
    )

    return parser.parse_args()


def print_configuration(args: argparse.Namespace) -> None:
    """
    Print the current configuration.

    Args:
        args: Parsed command-line arguments
    """
    logger.info("=== Test Configuration ===")
    logger.info(f"Model Types: {', '.join(args.model_types)}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Save Path: {args.save_path}")

    if args.cov_select_type:
        logger.info(f"Cov Select Type: {args.cov_select_type}")
    if args.filter_model:
        logger.info(f"Filter Model: {args.filter_model}")
    if args.skip_steps:
        logger.info(f"Skip Steps: {', '.join(map(str, args.skip_steps))}")
    logger.info("=" * 50)


def run_standard_workflow(testTool: CoverageTest, args: argparse.Namespace) -> None:
    """
    Run the standard testing workflow.

    Args:
        testTool: Configured CoverageTest instance
        args: Command-line arguments
    """
    # Step 2: Coverage detection on validation data
    if 2 not in args.skip_steps:
        logger.info("Step 2: Perform coverage detection on all val data")
        testTool.build_val_coverage_info()

    # Step 3: Select files
    if 3 not in args.skip_steps:
        logger.info(f"Step 3: Select {testTool.choice_samples} files (bottom/top/random)")
        testTool.select_files()

    # Step 4: Calculate coverage metrics
    if 4 not in args.skip_steps:
        logger.info(f"Step 4: Calculate coverage metrics for {testTool.choice_samples} samples")
        testTool.build_select_coverage_info()

    # Step 5: Calculate diversity
    if 5 not in args.skip_steps:
        logger.info(f"Step 5: Calculate diversity for {testTool.choice_samples} samples")
        testTool.cal_diversity_info()

    # Step 6: Analyze files
    if 6 not in args.skip_steps:
        import analyse

        analyse.analysis_iou(
            [],
            testTool.data_save_path_prefix,
            testTool.model_name,
            testTool.coverages_setting,
            cov_select_type=testTool.cov_select_type,
        )
        logger.info("Step 6: Analyze and summarize results")
        testTool.analyse_file()


def run_add_workflow(testTool: CoverageTest, args: argparse.Namespace) -> None:
    """
    Run the additive coverage workflow.

    Args:
        testTool: Configured CoverageTest instance
        args: Command-line arguments
    """
    # Step 2: Add metric coverage
    if 2 not in args.skip_steps:
        logger.info("Step 2: Add metric coverage")
        testTool.get_add_cover_info()


def run_diversity_workflow(testTool: CoverageTest, args: argparse.Namespace) -> None:
    """
    Run the diversity-focused testing workflow.

    Args:
        testTool: Configured CoverageTest instance
        args: Command-line arguments
    """
    # Step 2: Coverage detection
    if 2 not in args.skip_steps:
        logger.info("Step 2: Perform coverage detection on all val data")
        testTool.build_val_coverage_info()

    # Step 3: Select files with diversity considerations
    if 3 not in args.skip_steps:
        logger.info(f"Step 3: Select {testTool.choice_samples} files (bottom/top/random)")
        testTool.select_files()

    # Step 4: Calculate diversity metrics (extended)
    if 4 not in args.skip_steps:
        logger.info(f"Step 4: Calculate coverage metrics for {testTool.choice_samples} samples")
        testTool.build_select_coverage_info()

    # Step 5: Enhanced diversity calculation
    if 5 not in args.skip_steps:
        logger.info(f"Step 5: Calculate diversity for {testTool.choice_samples} samples")
        testTool.cal_diversity_info()

    # Step 6: Diversity analysis
    if 6 not in args.skip_steps:
        import analyse

        analyse.analysis_iou(
            [],
            testTool.data_save_path_prefix,
            testTool.model_name,
            testTool.coverages_setting,
            cov_select_type=testTool.cov_select_type,
        )
        logger.info("Step 6: Analyze and summarize results")
        testTool.analyse_file()


def get_coverage_settings(dataset: str, args: argparse.Namespace) -> dict:
    """
    Get coverage settings based on dataset and arguments.

    Args:
        dataset: Dataset name ('ade20k' or 'cityscapes')
        args: Command-line arguments

    Returns:
        Coverage settings dictionary
    """
    if args.coverage_settings:
        import json

        return json.loads(args.coverage_settings)

    # Default coverage settings
    base_cc_value = 19 if dataset == "cityscapes" else 150

    if args.mode == "add":
        return {
            "NC": [0.25, 0.5, 0.75],
            "KMNC": [25, 50, 100, 1000],
            "SNAC": [None],
            "NBC": [None],
            "TKNC": [5, 10, 15],
            "TKNP": [10, 25, 50],
            "CC": [5, 10, base_cc_value],
            "NLC": [None],
        }
    else:
        return None  # Use default from CoverageTest


def main():
    """
    Main execution function.
    """
    args = parse_arguments()

    # Print configuration
    print_configuration(args)

    # Map model types
    MODEL_TYPE = ["", "CNN", "Transformer", "Other"]
    model_type_indices = [MODEL_TYPE.index(mt) for mt in args.model_types]

    logger.info("Starting Coverage Testing Suite")

    for model_type_idx in model_type_indices:
        model_type = MODEL_TYPE[model_type_idx]
        logger.info(f"Model type: {model_type}")

        for dataset in args.datasets:
            model_infos, dataset_path_prefix = utility.get_file_device(
                dataset, model_type
            )

            for model_info in model_infos:
                model_name, config, checkpoint = (
                    model_info[0],
                    model_info[1],
                    model_info[2],
                )

                # Filter by specific model if requested
                if args.filter_model and model_name != args.filter_model:
                    continue

                logger.info("=" * 80)
                logger.info(f"Testing model: {model_name}")
                logger.info(f"Dataset: {dataset} | Type: {model_type}")
                logger.info("=" * 80)

                # Get coverage settings
                coverages_setting = get_coverage_settings(dataset, args)

                # Create test tool
                cov_select_type = args.cov_select_type if args.cov_select_type else None

                testTool = CoverageTest(
                    model_name,
                    dataset,
                    model_type,
                    config,
                    checkpoint,
                    dataset_path_prefix,
                    save_data_path=args.save_path,
                    cov_select_type=cov_select_type,
                    coverages_setting=coverages_setting,
                )

                utility.seed_everything(testTool.seed)
                logger.info(f"Starting test for {model_name}")

                # Run appropriate workflow
                try:
                    if args.mode == "add":
                        run_add_workflow(testTool, args)
                    elif args.mode == "diversity":
                        run_diversity_workflow(testTool, args)
                    else:
                        run_standard_workflow(testTool, args)

                    logger.info(f"Completed testing for {model_name}")

                except Exception as e:
                    logger.error(f"Error testing {model_name}: {e}")
                    import traceback

                    traceback.print_exc()

    logger.info("Testing Suite Completed")
    logger.info(f"Model Types: {', '.join(args.model_types)}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()