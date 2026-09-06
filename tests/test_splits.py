"""Split integrity tests: no sample in multiple splits, no artist leakage."""
from pathlib import Path

import pytest

from src.datasets import load_corrupted_track_ids, load_fma_metadata, load_fma_splits
from src.utils import load_config

METADATA_ROOT = Path("data/raw/fma_metadata")
requires_fma_metadata = pytest.mark.skipif(
    not (METADATA_ROOT / "tracks.csv").exists(), reason="FMA metadata not downloaded"
)


@requires_fma_metadata
def test_no_sample_in_multiple_splits():
    config = load_config("config.yaml")
    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(METADATA_ROOT, subset=subset)
    splits = load_fma_splits(tracks)  # raises AssertionError internally if violated
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids))


@requires_fma_metadata
def test_no_corrupted_tracks_in_splits():
    config = load_config("config.yaml")
    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(METADATA_ROOT, subset=subset)
    corrupted_ids = load_corrupted_track_ids()
    splits = load_fma_splits(tracks, exclude_track_ids=corrupted_ids)
    all_ids = set(splits["train"] + splits["val"] + splits["test"])
    assert not (all_ids & corrupted_ids)


@requires_fma_metadata
def test_no_artist_overlap_between_train_and_test():
    config = load_config("config.yaml")
    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(METADATA_ROOT, subset=subset)
    splits = load_fma_splits(tracks)
    train_artists = set(tracks[tracks["track_id"].isin(splits["train"])]["artist_id"])
    test_artists = set(tracks[tracks["track_id"].isin(splits["test"])]["artist_id"])
    assert not (train_artists & test_artists)
