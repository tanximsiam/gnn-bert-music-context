"""Phase 0 basic tests: reproducible seeding and config loading."""
from __future__ import annotations

import random

import numpy as np

from src.utils import load_config, set_seed


def test_config_loads_required_sections():
    config = load_config("config.yaml")
    for section in ("dataset", "audio", "graph", "bert", "gnn", "fusion", "training"):
        assert section in config, f"missing config section: {section}"


def test_set_seed_is_deterministic():
    set_seed(123)
    a = [random.random() for _ in range(5)], np.random.rand(5).tolist()
    set_seed(123)
    b = [random.random() for _ in range(5)], np.random.rand(5).tolist()
    assert a == b
