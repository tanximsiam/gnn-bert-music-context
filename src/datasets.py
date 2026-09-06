"""Dataset loading, alignment, and split logic for FMA, DEAM, and MusicCaps.

Implemented in Phase 1 (FMA metadata/audio + verified splits). DEAM (Task 3
emotion aux) and MusicCaps (Task 4) loaders are added later and gated behind
explicit user approval before any download occurs, per project data rules.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def load_fma_metadata(metadata_root: str | Path, subset: str) -> pd.DataFrame:
    """Load tracks.csv and restrict to the given FMA subset ('small' or 'medium').

    'medium' includes all 'small' tracks (FMA subsets are nested), matching the
    official FMA metadata convention.
    """
    metadata_root = Path(metadata_root)
    tracks_path = metadata_root / "tracks.csv"
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.csv not found under {metadata_root}")

    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    if subset == "small":
        allowed = {"small"}
    elif subset == "medium":
        allowed = {"small", "medium"}
    else:
        raise ValueError(f"Unsupported FMA subset: {subset!r}")

    subset_df = tracks[tracks[("set", "subset")].isin(allowed)]

    df = pd.DataFrame(
        {
            "track_id": subset_df.index,
            "genre_top": subset_df[("track", "genre_top")].values,
            "artist_id": subset_df[("artist", "id")].values,
            "artist_bio": subset_df[("artist", "bio")].values,
            "official_split": subset_df[("set", "split")].values,
            "track_tags": subset_df[("track", "tags")].apply(_parse_tag_list).values,
            "artist_tags": subset_df[("artist", "tags")].apply(_parse_tag_list).values,
        }
    )

    df = df.dropna(subset=["genre_top"])
    return df.reset_index(drop=True)


def _parse_tag_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def load_fma_splits(
    tracks_df: pd.DataFrame, exclude_track_ids: set[int] | None = None
) -> dict[str, list[int]]:
    """Build train/val/test track-ID lists from FMA's official split column.

    `exclude_track_ids` drops known-corrupted/undecodable tracks (see
    data/splits/fma_medium_corrupted_track_ids.json) from every split before
    the leakage checks run.

    Verifies no track appears in more than one split and no artist appears in
    more than one split (data leakage rules #4). Raises AssertionError if
    either check fails.
    """
    exclude_track_ids = exclude_track_ids or set()
    split_name_map = {"training": "train", "validation": "val", "test": "test"}
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    artist_by_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}

    for official_name, out_name in split_name_map.items():
        subset = tracks_df[
            (tracks_df["official_split"] == official_name)
            & (~tracks_df["track_id"].isin(exclude_track_ids))
        ]
        splits[out_name] = subset["track_id"].tolist()
        artist_by_split[out_name] = set(subset["artist_id"].tolist())

    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids)), "a track_id appears in more than one split"

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = artist_by_split[a] & artist_by_split[b]
        assert not overlap, f"artist leakage between {a} and {b}: {len(overlap)} shared artist_ids"

    return splits


def save_splits(splits: dict[str, list[int]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(splits, f, indent=2)


def load_corrupted_track_ids(
    path: str | Path = "data/splits/fma_medium_corrupted_track_ids.json",
) -> set[int]:
    """Load the set of known-undecodable FMA track IDs to exclude from splits."""
    path = Path(path)
    if not path.exists():
        return set()
    with path.open() as f:
        return set(json.load(f)["corrupted_track_ids"])


def compute_dataset_statistics(tracks_df: pd.DataFrame) -> dict[str, Any]:
    """Summary statistics for dataset-card / EDA reporting (no leakage-sensitive info)."""
    tagged = tracks_df["track_tags"].apply(len) > 0
    return {
        "num_tracks": int(len(tracks_df)),
        "num_artists": int(tracks_df["artist_id"].nunique()),
        "num_genres": int(tracks_df["genre_top"].nunique()),
        "genre_counts": tracks_df["genre_top"].value_counts().to_dict(),
        "split_counts": tracks_df["official_split"].value_counts().to_dict(),
        "num_tracks_with_track_tags": int(tagged.sum()),
        "num_tracks_with_artist_bio": int(tracks_df["artist_bio"].notna().sum()),
    }


def load_deam(*args, **kwargs):
    raise NotImplementedError(
        "Optional emotion extension (Task 3). Requires explicit user approval to download DEAM."
    )


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text).strip()


def build_task1_top_tags(tracks_df: pd.DataFrame, top_k: int = 20) -> list[str]:
    """Most frequent track-level tags across the given tracks, used as the
    fixed multi-label target vocabulary for Task 1."""
    counter: Counter[str] = Counter()
    for tags in tracks_df["track_tags"]:
        counter.update(t.lower() for t in tags)
    return [tag for tag, _ in counter.most_common(top_k)]


def build_task1_dataset(tracks_df: pd.DataFrame, top_tags: list[str]) -> pd.DataFrame:
    """Build the Task 1 multi-label tag dataset.

    Restricted to tracks with at least one track_tag (the only tracks with a
    genuine tag signal). Input text = raw artist bio (HTML-stripped, unmasked).
    Falls back to a placeholder string when no bio is available.
    """
    tagged = tracks_df[tracks_df["track_tags"].apply(len) > 0].copy()

    def make_row(row: pd.Series) -> pd.Series:
        bio = _strip_html(row["artist_bio"]) if isinstance(row["artist_bio"], str) else ""
        if not bio:
            bio = "No artist description available."
        labels = [1 if tag in row["track_tags_lower"] else 0 for tag in top_tags]
        return pd.Series({"track_id": row["track_id"], "text": bio, "labels": labels})

    tagged["track_tags_lower"] = tagged["track_tags"].apply(lambda tags: {t.lower() for t in tags})
    return tagged.apply(make_row, axis=1)



def load_musiccaps(*args, **kwargs):
    raise NotImplementedError(
        "Optional Task 4 extension. Requires explicit user approval to download MusicCaps."
    )
