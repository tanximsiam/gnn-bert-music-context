"""Baselines B1 (majority/random tag predictor) and optional B4 (PCA + MLP on
hand-crafted audio features), per spec section 8.
"""
from __future__ import annotations

import numpy as np


def majority_random_baseline(
    train_labels: list[int], num_samples: int, mode: str = "majority", seed: int = 42
) -> np.ndarray:
    """B1: predict either the most frequent train-split class (mode='majority')
    or a uniform-random class (mode='random') for every test sample."""
    train_labels = np.asarray(train_labels)
    if mode == "majority":
        majority_class = np.bincount(train_labels).argmax()
        return np.full(num_samples, majority_class, dtype=np.int64)
    if mode == "random":
        rng = np.random.RandomState(seed)
        classes = np.unique(train_labels)
        return rng.choice(classes, size=num_samples)
    raise ValueError(f"Unsupported mode: {mode!r}")


class PcaMlpBaseline:
    def __init__(self, n_components: int, hidden_dim: int, num_labels: int):
        raise NotImplementedError("Optional B4 baseline: PCA + MLP on hand-crafted features.")
