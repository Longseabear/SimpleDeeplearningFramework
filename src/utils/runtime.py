import platform
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torchvision


def collect_runtime_info() -> dict[str, str]:
    try:
        original_cwd = hydra.utils.get_original_cwd()
    except ValueError:
        original_cwd = str(Path.cwd())

    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "original_cwd": original_cwd,
    }


def save_runtime_info(output_path: Path, runtime_info: dict[str, str]) -> None:
    lines = [f"{key}: {value}" for key, value in runtime_info.items()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
