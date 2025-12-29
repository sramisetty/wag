"""
WAG LLM Fine-Tuning Utilities
==============================
Shared utility functions for the WAG training pipeline

Author: Enterprise Architecture Team
Created: November 2025
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def setup_logging(
    name: str = "wag",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for log output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_device_info() -> Dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'platform': sys.platform,
        'python_version': sys.version,
        'cuda_available': False,
        'cuda_version': None,
        'gpu_count': 0,
        'gpus': []
    }

    try:
        import torch

        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()

        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()

            for i in range(info['gpu_count']):
                gpu_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
                }
                info['gpus'].append(gpu_info)

    except ImportError:
        info['torch_version'] = 'not installed'

    return info


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_project_root() -> Path:
    """Get the project root directory (WAG folder)."""
    # Navigate up from utils to scripts to WAG
    return Path(__file__).parent.parent.parent


def ensure_directories() -> Dict[str, Path]:
    """
    Ensure all required output directories exist.

    Returns:
        Dictionary of directory paths
    """
    root = get_project_root()
    scripts_dir = root / "scripts"

    directories = {
        'output': scripts_dir / 'output',
        'data': scripts_dir / 'output' / 'data',
        'data_enriched': scripts_dir / 'output' / 'data_enriched',
        'models': scripts_dir / 'output' / 'models',
        'checkpoints': scripts_dir / 'output' / 'checkpoints',
        'reports': scripts_dir / 'output' / 'reports',
        'logs': scripts_dir / 'output' / 'logs',
    }

    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def print_banner(title: str, width: int = 60) -> None:
    """Print a formatted banner."""
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} completed in {elapsed}")

    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
