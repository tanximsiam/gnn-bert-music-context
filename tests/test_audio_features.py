"""Audio preprocessing tests: shapes, determinism, and the no-NaN/Inf
validation check required by the project's automated data-quality rules.
"""
from pathlib import Path

import numpy as np
import pytest

from src.audio_features import extract_segment_features, load_audio, segment_audio

FMA_ROOT = Path("data/raw/fma_medium")
requires_fma_audio = pytest.mark.skipif(not FMA_ROOT.exists(), reason="FMA audio not downloaded")


def _sample_track_path() -> Path:
    return next(FMA_ROOT.rglob("*.mp3"))


@requires_fma_audio
def test_load_audio_is_resampled_and_peak_normalized():
    waveform = load_audio(str(_sample_track_path()), sample_rate=22050)
    assert waveform.ndim == 1
    assert np.abs(waveform).max() <= 1.0 + 1e-6


@requires_fma_audio
def test_segment_audio_fixed_length_and_full_coverage():
    waveform = load_audio(str(_sample_track_path()), sample_rate=22050)
    segments = segment_audio(waveform, sample_rate=22050, segment_seconds=5.0)
    window = int(5.0 * 22050)
    assert all(len(s) == window for s in segments)
    assert sum(len(s) for s in segments) >= len(waveform)


@requires_fma_audio
def test_segment_features_no_nan_or_inf():
    waveform = load_audio(str(_sample_track_path()), sample_rate=22050)
    segments = segment_audio(waveform, sample_rate=22050, segment_seconds=5.0)
    for segment in segments:
        features = extract_segment_features(segment, sample_rate=22050, n_mfcc=20, use_chroma=True)
        assert np.isfinite(features).all(), "segment features contain NaN/Inf"


@requires_fma_audio
def test_segment_feature_dimension_matches_config():
    waveform = load_audio(str(_sample_track_path()), sample_rate=22050)
    segment = segment_audio(waveform, sample_rate=22050, segment_seconds=5.0)[0]
    features = extract_segment_features(segment, sample_rate=22050, n_mfcc=20, use_chroma=True)
    assert features.shape == (20 * 2 + 12 * 2,)  # MFCC mean+std, chroma mean+std
