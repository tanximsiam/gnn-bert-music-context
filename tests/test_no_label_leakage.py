"""Data-leakage guard: target tag/genre must never appear in the BERT input
text.
"""
import re
from pathlib import Path

import pytest

from src.datasets import (
    build_task1_dataset,
    build_task1_top_tags,
    load_corrupted_track_ids,
    load_fma_metadata,
    load_fma_splits,
)

METADATA_ROOT = Path("data/raw/fma_metadata")
requires_fma_metadata = pytest.mark.skipif(
    not (METADATA_ROOT / "tracks.csv").exists(), reason="FMA metadata not downloaded"
)


@requires_fma_metadata
def test_target_tag_not_present_in_bert_input_text():
    tracks = load_fma_metadata(METADATA_ROOT, subset="medium")
    splits = load_fma_splits(tracks, exclude_track_ids=load_corrupted_track_ids())
    top_tags = build_task1_top_tags(tracks[tracks["track_id"].isin(splits["train"])], top_k=20)
    task1_df = build_task1_dataset(tracks, top_tags)
    tracks_by_id = tracks.set_index("track_id")

    violations = []
    for _, row in task1_df.head(500).iterrows():
        genre = tracks_by_id.loc[row["track_id"], "genre_top"]
        for word in top_tags + [genre]:
            if re.search(r"\b" + re.escape(word) + r"\b", row["text"], flags=re.IGNORECASE):
                violations.append((row["track_id"], word))
    assert not violations, f"leaked words found in BERT input text: {violations[:5]}"


@requires_fma_metadata
def test_only_remaining_contextual_tags_passed_to_bert():
    """The 20 target tags are masked from text regardless of whether they apply
    to the given track; only non-target information (bio prose) remains."""
    tracks = load_fma_metadata(METADATA_ROOT, subset="medium")
    splits = load_fma_splits(tracks, exclude_track_ids=load_corrupted_track_ids())
    top_tags = build_task1_top_tags(tracks[tracks["track_id"].isin(splits["train"])], top_k=20)
    task1_df = build_task1_dataset(tracks, top_tags)
    assert "[MASK]" in " ".join(task1_df["text"].head(50)), "expected at least some masking to occur"
