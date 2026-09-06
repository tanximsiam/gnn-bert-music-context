"""Segment graph construction: nodes = audio segments, edges = temporal
adjacency (i <-> i+1) plus cosine-similarity edges (cosine(f_i, f_j) > tau).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from src.audio_features import extract_segment_features, load_audio, segment_audio


def _cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    # Z-score each feature dimension across this track's own segments (no cross-sample
    # leakage) before computing similarity. Without this, the large-magnitude MFCC
    # log-energy coefficient dominates the vector norm and cosine similarity collapses
    # to ~1.0 for nearly every segment pair, making tau meaningless.
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    standardized = (features - mean) / std

    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = standardized / norms
    return normalized @ normalized.T


def build_segment_graph(
    node_features: np.ndarray,
    similarity_threshold: float,
    bidirectional: bool = True,
    self_loops: bool = False,
) -> dict:
    """Build a segment graph from a (num_segments, feature_dim) feature matrix.

    Returns a dict with:
      - x: (N, D) float32 node features
      - edge_index: (2, E) int64 array of [source, target] edge endpoints
      - edge_type: (E,) array, "temporal" or "similarity", for inspection/debugging
    """
    num_nodes = node_features.shape[0]
    if num_nodes == 0:
        raise ValueError("node_features must contain at least one segment")

    edges: list[tuple[int, int]] = []
    edge_types: list[str] = []

    # A. Temporal adjacency: i <-> i+1
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
        edge_types.append("temporal")
        if bidirectional:
            edges.append((i + 1, i))
            edge_types.append("temporal")

    # B. Similarity edges: cosine(feature_i, feature_j) > tau
    sim = _cosine_similarity_matrix(node_features)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if not bidirectional and j < i:
                continue
            if sim[i, j] > similarity_threshold:
                edges.append((i, j))
                edge_types.append("similarity")

    if self_loops:
        for i in range(num_nodes):
            edges.append((i, i))
            edge_types.append("self_loop")

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return {
        "x": node_features.astype(np.float32),
        "edge_index": edge_index,
        "edge_type": np.array(edge_types),
    }


def visualize_graph(graph: dict, out_path: str, title: str = "") -> None:
    """Render a segment graph (temporal edges vs. similarity edges) to a PNG."""
    import matplotlib.pyplot as plt
    import networkx as nx

    num_nodes = graph["x"].shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    for (src, dst), etype in zip(graph["edge_index"].T, graph["edge_type"]):
        g.add_edge(int(src), int(dst), etype=str(etype))

    pos = {i: (i, 0) for i in range(num_nodes)}
    temporal_edges = [(u, v) for u, v, d in g.edges(data=True) if d["etype"] == "temporal"]
    similarity_edges = [(u, v) for u, v, d in g.edges(data=True) if d["etype"] == "similarity"]

    plt.figure(figsize=(max(6, num_nodes), 3))
    nx.draw_networkx_nodes(g, pos, node_size=300, node_color="lightblue")
    nx.draw_networkx_labels(g, pos, font_size=8)
    nx.draw_networkx_edges(g, pos, edgelist=temporal_edges, edge_color="black", connectionstyle="arc3,rad=0.0")
    nx.draw_networkx_edges(
        g, pos, edgelist=similarity_edges, edge_color="red", connectionstyle="arc3,rad=0.3", style="dashed"
    )
    plt.title(title or "Segment graph (black=temporal, red dashed=similarity)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def track_audio_path(track_id: int, root: str | Path) -> Path:
    """FMA convention: mp3s live under root/<first 3 digits of zero-padded id>/<id>.mp3"""
    tid_str = f"{track_id:06d}"
    return Path(root) / tid_str[:3] / f"{tid_str}.mp3"


def build_genre_label_map(tracks_df: pd.DataFrame) -> dict[str, int]:
    """Alphabetical genre -> integer label mapping, fixed regardless of split."""
    genres = sorted(tracks_df["genre_top"].unique())
    return {genre: idx for idx, genre in enumerate(genres)}


def build_or_load_track_graph(
    track_id: int,
    audio_root: str | Path,
    cache_dir: str | Path,
    sample_rate: int,
    segment_seconds: float,
    n_mfcc: int,
    use_chroma: bool,
    similarity_threshold: float,
    bidirectional: bool,
    self_loops: bool,
) -> Data:
    """Build a track's segment graph (or load it from cache if already built)."""
    cache_path = Path(cache_dir) / f"{track_id:06d}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    path = track_audio_path(track_id, audio_root)
    waveform = load_audio(str(path), sample_rate=sample_rate)
    segments = segment_audio(waveform, sample_rate, segment_seconds)
    features = np.stack(
        [extract_segment_features(s, sample_rate, n_mfcc, use_chroma) for s in segments]
    )
    graph = build_segment_graph(features, similarity_threshold, bidirectional, self_loops)

    data = Data(x=torch.tensor(graph["x"]), edge_index=torch.tensor(graph["edge_index"]))
    data.track_id = track_id
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data


class SegmentGraphDataset(Dataset):
    """Lazily builds (and caches to disk) segment graphs for a list of track IDs,
    labeling each with its genre_top index from `label_map`."""

    def __init__(
        self,
        track_ids: list[int],
        tracks_df: pd.DataFrame,
        audio_root: str | Path,
        cache_dir: str | Path,
        label_map: dict[str, int],
        audio_config: dict[str, Any],
        graph_config: dict[str, Any],
        sample_rate: int,
    ):
        super().__init__()
        self.track_ids = track_ids
        self.id_to_genre = dict(zip(tracks_df["track_id"], tracks_df["genre_top"]))
        self.audio_root = audio_root
        self.cache_dir = cache_dir
        self.label_map = label_map
        self.audio_config = audio_config
        self.graph_config = graph_config
        self.sample_rate = sample_rate

    def len(self) -> int:
        return len(self.track_ids)

    def get(self, idx: int) -> Data:
        track_id = self.track_ids[idx]
        data = build_or_load_track_graph(
            track_id,
            self.audio_root,
            self.cache_dir,
            self.sample_rate,
            self.audio_config["segment_seconds"],
            self.audio_config["n_mfcc"],
            self.audio_config["use_chroma"],
            self.graph_config["similarity_threshold"],
            self.graph_config["bidirectional_edges"],
            self.graph_config["self_loops"],
        ).clone()
        genre = self.id_to_genre[track_id]
        data.y = torch.tensor([self.label_map[genre]], dtype=torch.long)
        return data
