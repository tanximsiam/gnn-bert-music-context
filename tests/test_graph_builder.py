"""Segment graph construction tests."""
import numpy as np

from src.graph_builder import build_segment_graph


def test_temporal_edges_connect_adjacent_segments():
    features = np.random.RandomState(0).rand(5, 8)
    graph = build_segment_graph(features, similarity_threshold=2.0)  # tau=2 disables similarity edges
    temporal = graph["edge_index"][:, graph["edge_type"] == "temporal"]
    pairs = set(map(tuple, temporal.T.tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)}


def test_similarity_edges_respect_tau_threshold():
    # two identical feature rows (cosine sim = 1) should get a similarity edge at any tau < 1
    features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = build_segment_graph(features, similarity_threshold=0.99)
    sim_pairs = set(map(tuple, graph["edge_index"][:, graph["edge_type"] == "similarity"].T.tolist()))
    assert (0, 1) in sim_pairs and (1, 0) in sim_pairs
    assert (0, 2) not in sim_pairs and (1, 2) not in sim_pairs


def test_edge_indices_are_within_node_bounds():
    features = np.random.RandomState(1).rand(6, 4)
    graph = build_segment_graph(features, similarity_threshold=0.5)
    num_nodes = graph["x"].shape[0]
    assert graph["edge_index"].min() >= 0
    assert graph["edge_index"].max() < num_nodes


def test_self_loops_optional():
    features = np.random.RandomState(2).rand(3, 4)
    without = build_segment_graph(features, similarity_threshold=2.0, self_loops=False)
    with_loops = build_segment_graph(features, similarity_threshold=2.0, self_loops=True)
    assert "self_loop" not in without["edge_type"]
    assert list(with_loops["edge_type"]).count("self_loop") == 3
