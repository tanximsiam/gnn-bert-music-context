"""Audio loading, resampling, per-track normalization, segmentation, and node
feature extraction (chroma + MFCC statistics).
"""
from __future__ import annotations

import librosa
import numpy as np


def load_audio(path: str, sample_rate: int = 22050) -> np.ndarray:
    """Load an audio file, resampled to `sample_rate`, peak-normalized to [-1, 1]."""
    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def segment_audio(waveform: np.ndarray, sample_rate: int, segment_seconds: float = 5.0) -> list[np.ndarray]:
    """Split a waveform into fixed-length windows of `segment_seconds`.

    The final partial window (if any) is zero-padded to full length so no
    audio is discarded and every segment has equal length for batching.
    """
    window = int(segment_seconds * sample_rate)
    if window <= 0:
        raise ValueError("segment_seconds * sample_rate must be positive")

    segments = []
    for start in range(0, len(waveform), window):
        chunk = waveform[start : start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        segments.append(chunk)
    return segments


def extract_segment_features(
    segment: np.ndarray, sample_rate: int, n_mfcc: int = 20, use_chroma: bool = True
) -> np.ndarray:
    """Extract a fixed-size node-feature vector for one segment.

    Features are [MFCC mean, MFCC std] (2*n_mfcc dims), optionally concatenated
    with [chroma mean, chroma std] (24 dims for the standard 12 chroma bins).
    Using summary statistics (not raw spectrograms) keeps per-segment features
    small, per the "don't save huge raw spectrogram tensors" project rule.
    """
    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
    features = [mfcc.mean(axis=1), mfcc.std(axis=1)]

    if use_chroma:
        chroma = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
        features += [chroma.mean(axis=1), chroma.std(axis=1)]

    return np.concatenate(features).astype(np.float32)


def extract_fixed_length_melspec(
    waveform: np.ndarray, sample_rate: int, n_mels: int = 128, target_seconds: float = 30.0
) -> np.ndarray:
    """Log-mel spectrogram for the CNN baseline (B2), cropped/zero-padded to a
    fixed `target_seconds` so every track produces the same (n_mels, T) shape
    for batching. Only computed for the CNN baseline, never cached wholesale
    for the GNN pipeline (per the "don't save huge raw spectrogram tensors"
    rule) — the CNN baseline is the one place a full spectrogram is required.
    """
    target_len = int(target_seconds * sample_rate)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]

    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

