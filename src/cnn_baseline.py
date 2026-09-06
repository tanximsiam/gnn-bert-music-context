"""CNN baseline (B2) on log-mel spectrograms, per Task 2 required comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.audio_features import extract_fixed_length_melspec, load_audio
from src.graph_builder import track_audio_path


class MelSpecCNN(nn.Module):
    """Small 4-block conv net (conv-bn-relu-maxpool) + global average pool + linear
    head, operating directly on a fixed-size (1, n_mels, T) log-mel spectrogram."""

    def __init__(self, n_mels: int, num_classes: int):
        super().__init__()
        channels = [1, 16, 32, 64, 128]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x)  # logits; use CrossEntropyLoss


def build_or_load_melspec(
    track_id: int, audio_root: str | Path, cache_dir: str | Path, sample_rate: int, n_mels: int, target_seconds: float
) -> np.ndarray:
    """Build a track's fixed-size log-mel spectrogram, cached to disk as .npy."""
    cache_path = Path(cache_dir) / f"{track_id:06d}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    path = track_audio_path(track_id, audio_root)
    waveform = load_audio(str(path), sample_rate=sample_rate)
    mel = extract_fixed_length_melspec(waveform, sample_rate, n_mels, target_seconds)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mel)
    return mel


class MelSpecDataset(Dataset):
    """Lazily builds (and caches) fixed-size log-mel spectrograms for a list of
    track IDs, labeling each with its genre_top index from `label_map`."""

    def __init__(
        self,
        track_ids: list[int],
        tracks_df: pd.DataFrame,
        audio_root: str | Path,
        cache_dir: str | Path,
        label_map: dict[str, int],
        sample_rate: int,
        n_mels: int,
        target_seconds: float = 30.0,
    ):
        self.track_ids = track_ids
        self.id_to_genre = dict(zip(tracks_df["track_id"], tracks_df["genre_top"]))
        self.audio_root = audio_root
        self.cache_dir = cache_dir
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_seconds = target_seconds

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        track_id = self.track_ids[idx]
        mel = build_or_load_melspec(
            track_id, self.audio_root, self.cache_dir, self.sample_rate, self.n_mels, self.target_seconds
        )
        label = self.label_map[self.id_to_genre[track_id]]
        return {"x": torch.tensor(mel).unsqueeze(0), "y": torch.tensor(label, dtype=torch.long)}
