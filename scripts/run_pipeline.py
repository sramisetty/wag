"""
WAG Full Pipeline Runner
=========================
Runs the complete data preparation, training, and evaluation pipeline

Usage:
    # Full pipeline
    python run_pipeline.py --all

    # Data preparation only
    python run_pipeline.py --prep

    # Training only (assumes data is ready)
    python run_pipeline.py --train

    # Evaluation only (assumes model is trained)
    python run_pipeline.py --eval

Author: Enterprise Architecture Team
Created: November 2025
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.helpers import setup_logging, get_device_info, print_banner, Timer, ensure_directories

# Setup logging
logger = setup_logging("pipeline")


def run_command(cmd: list, cwd: str = None, description: str = "Running command"):
    """Run a command and handle errors."""
    logger.info(f"{description}...")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return False


def check_prerequisites():
    """Check that all prerequisites are met."""
    logger.info("Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 10):
        logger.warning("Python 3.10+ recommended")

    # Check for required files
    wag_dir = SCRIPTS_DIR.parent
    required_files = [
        wag_dir / "WAG History.xlsx",
        wag_dir / "WAG Master Item.xlsx",
    ]

    missing = []
    for f in required_files:
        if not f.exists():
            missing.append(f.name)

    if missing:
        logger.error(f"Missing required files: {', '.join(missing)}")
        return False

    # Check GPU
    device_info = get_device_info()
    if device_info['cuda_available']:
        logger.info(f"GPU available: {device_info['gpus'][0]['name']}")
        logger.info(f"GPU memory: {device_info['gpus'][0]['memory_total_gb']:.1f} GB")
    else:
        logger.warning("No GPU detected. Training will be very slow!")

    return True


def run_data_preparation():
    """Run data preparation pipeline."""
    print_banner("DATA PREPARATION")

    data_prep_dir = SCRIPTS_DIR / "data_prep"
    wag_dir = SCRIPTS_DIR.parent

    with Timer("Data extraction"):
        success = run_command(
            [
                sys.executable,
                "extract_training_data.py",
                "--input", str(wag_dir / "WAG History.xlsx"),
                "--output", str(SCRIPTS_DIR / "output" / "data")
            ],
            cwd=str(data_prep_dir),
            description="Extracting training data from WAG History"
        )
        if not success:
            return False

    with Timer("Product enrichment"):
        success = run_command(
            [
                sys.executable,
                "enrich_with_products.py",
                "--training", str(SCRIPTS_DIR / "output" / "data" / "wag_training_data_raw.json"),
                "--master", str(wag_dir / "WAG Master Item.xlsx"),
                "--output", str(SCRIPTS_DIR / "output" / "data_enriched")
            ],
            cwd=str(data_prep_dir),
            description="Enriching data with product details"
        )
        if not success:
            logger.warning("Product enrichment failed, continuing with basic data")

    with Timer("Data validation"):
        success = run_command(
            [
                sys.executable,
                "validate_data.py",
                "--input", str(SCRIPTS_DIR / "output" / "data" / "wag_training_data_raw.json"),
                "--output", str(SCRIPTS_DIR / "output" / "reports" / "validation_report.md")
            ],
            cwd=str(data_prep_dir),
            description="Validating data quality"
        )

    logger.info("Data preparation complete!")
    return True


def run_training():
    """Run model training."""
    print_banner("MODEL TRAINING")

    training_dir = SCRIPTS_DIR / "training"

    # Check if training data exists
    train_file = SCRIPTS_DIR / "output" / "data_enriched" / "wag_enriched_train.jsonl"
    if not train_file.exists():
        train_file = SCRIPTS_DIR / "output" / "data" / "wag_train.jsonl"

    if not train_file.exists():
        logger.error("Training data not found. Run data preparation first.")
        return False

    with Timer("Model training"):
        success = run_command(
            [
                sys.executable,
                "train.py",
                "--config", "config.yaml"
            ],
            cwd=str(training_dir),
            description="Fine-tuning model with QLoRA"
        )
        if not success:
            return False

    logger.info("Training complete!")
    return True


def run_evaluation():
    """Run model evaluation."""
    print_banner("MODEL EVALUATION")

    training_dir = SCRIPTS_DIR / "training"
    model_dir = SCRIPTS_DIR / "output" / "models" / "wag-copywriter"

    if not model_dir.exists():
        logger.error("Trained model not found. Run training first.")
        return False

    # Find test file
    test_file = SCRIPTS_DIR / "output" / "data_enriched" / "wag_enriched_test.jsonl"
    if not test_file.exists():
        test_file = SCRIPTS_DIR / "output" / "data" / "wag_test.jsonl"

    with Timer("Model evaluation"):
        success = run_command(
            [
                sys.executable,
                "evaluate.py",
                "--model", str(model_dir),
                "--test", str(test_file),
                "--num-samples", "100",  # Evaluate on subset for speed
                "--output", str(SCRIPTS_DIR / "output" / "reports" / "evaluation")
            ],
            cwd=str(training_dir),
            description="Evaluating model performance"
        )

    logger.info("Evaluation complete!")
    return success


def run_ollama_setup():
    """Setup Ollama for deployment."""
    print_banner("OLLAMA SETUP")

    inference_dir = SCRIPTS_DIR / "inference"

    # First try to setup base model
    success = run_command(
        [
            sys.executable,
            "ollama_setup.py",
            "--setup-base"
        ],
        cwd=str(inference_dir),
        description="Setting up Ollama base model"
    )

    # Then try to create fine-tuned model if available
    model_dir = SCRIPTS_DIR / "output" / "models" / "wag-copywriter"
    if model_dir.exists():
        run_command(
            [
                sys.executable,
                "ollama_setup.py",
                "--create"
            ],
            cwd=str(inference_dir),
            description="Creating fine-tuned Ollama model"
        )

    return success


def main():
    parser = argparse.ArgumentParser(
        description='Run WAG LLM fine-tuning pipeline'
    )

    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline (prep + train + eval)')
    parser.add_argument('--prep', action='store_true',
                        help='Run data preparation only')
    parser.add_argument('--train', action='store_true',
                        help='Run training only')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation only')
    parser.add_argument('--ollama', action='store_true',
                        help='Setup Ollama deployment')
    parser.add_argument('--check', action='store_true',
                        help='Check prerequisites only')

    args = parser.parse_args()

    # Default to showing help if no args
    if not any([args.all, args.prep, args.train, args.eval, args.ollama, args.check]):
        parser.print_help()
        print("\nExamples:")
        print("  python run_pipeline.py --all     # Run full pipeline")
        print("  python run_pipeline.py --prep    # Data preparation only")
        print("  python run_pipeline.py --train   # Training only")
        print("  python run_pipeline.py --eval    # Evaluation only")
        print("  python run_pipeline.py --ollama  # Setup Ollama")
        return

    # Banner
    print_banner("WAG AD COPY LLM FINE-TUNING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Ensure directories
    ensure_directories()

    # Check prerequisites
    if not check_prerequisites():
        if not args.check:
            logger.error("Prerequisites check failed. Aborting.")
            sys.exit(1)
        return

    if args.check:
        logger.info("Prerequisites check passed!")
        return

    # Run pipeline stages
    total_timer = Timer("Total pipeline")
    total_timer.__enter__()

    success = True

    if args.all or args.prep:
        if not run_data_preparation():
            success = False
            if not args.all:
                sys.exit(1)

    if args.all or args.train:
        if not run_training():
            success = False
            if not args.all:
                sys.exit(1)

    if args.all or args.eval:
        if not run_evaluation():
            success = False

    if args.ollama:
        run_ollama_setup()

    total_timer.__exit__(None, None, None)

    # Summary
    print()
    print_banner("PIPELINE COMPLETE" if success else "PIPELINE COMPLETED WITH ERRORS")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if success:
        print("\nNext steps:")
        print("  1. Review validation report: scripts/output/reports/validation_report.md")
        print("  2. Review evaluation report: scripts/output/reports/evaluation.md")
        print("  3. Test generation: python scripts/inference/generate.py --wics '691500' --price '$9.99'")
        print("  4. Setup Ollama: python scripts/run_pipeline.py --ollama")


if __name__ == '__main__':
    main()
