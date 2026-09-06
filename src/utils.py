"""Shared utilities: reproducible seeding, config loading, logging, run directories."""
from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML experiment config into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def make_run_dir(base_dir: str | Path, task_name: str) -> Path:
    """Create a unique, never-overwritten run directory under base_dir/task_name/."""
    base = Path(base_dir) / task_name
    base.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base / run_id
    suffix = 1
    original = run_dir
    while run_dir.exists():
        run_dir = Path(f"{original}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def save_run_metadata(run_dir: str | Path, config: dict[str, Any], split_name: str) -> None:
    """Record the resolved config, dataset split, and git commit for an experiment run."""
    metadata = {
        "config": config,
        "split": split_name,
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with (Path(run_dir) / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return a configured logger that writes to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
