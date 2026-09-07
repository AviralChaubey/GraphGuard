import numpy as np
import pytest

from src.explain.evidence import build_evidence_subgraph


def _star_edge_index():
    # target=0, connected to 1..4; plus an unrelated edge 5-6 far from the target.
    src = [0, 0, 0, 0, 5]
    dst = [1, 2, 3, 4, 6]
    return np.array([src, dst])


def test_keeps_only_edges_touching_target():
    edge_index = _star_edge_index()
    edge_mask = np.array([0.9, 0.8, 0.7, 0.6, 0.95])  # edge 4 (5-6) has the highest weight
    ev = build_evidence_subgraph(edge_index, edge_mask, target_idx=0, top_k_edges=10, edge_threshold=0.0)

    assert (5, 6) not in ev.edges and (6, 5) not in ev.edges
    assert set(ev.edges) == {(0, 1), (0, 2), (0, 3), (0, 4)}
    assert set(ev.nodes) == {0, 1, 2, 3, 4}


def test_respects_top_k_and_ordering():
    edge_index = _star_edge_index()
    edge_mask = np.array([0.9, 0.8, 0.7, 0.6, 0.0])
    ev = build_evidence_subgraph(edge_index, edge_mask, target_idx=0, top_k_edges=2, edge_threshold=0.0)

    assert len(ev.edges) == 2
    assert ev.edges == [(0, 1), (0, 2)]  # highest-weight edges first
    assert ev.edge_weights == [0.9, 0.8]


def test_edge_threshold_drops_weak_edges():
    edge_index = _star_edge_index()
    edge_mask = np.array([0.9, 0.05, 0.05, 0.05, 0.0])
    ev = build_evidence_subgraph(edge_index, edge_mask, target_idx=0, top_k_edges=10, edge_threshold=0.1)

    assert ev.edges == [(0, 1)]


def test_never_returns_completely_empty_when_edges_exist():
    edge_index = _star_edge_index()
    edge_mask = np.array([0.01, 0.01, 0.01, 0.01, 0.0])
    ev = build_evidence_subgraph(edge_index, edge_mask, target_idx=0, top_k_edges=10, edge_threshold=0.5)

    assert len(ev.edges) == 1  # falls back to the strongest edge rather than nothing


def test_mismatched_lengths_raise():
    edge_index = _star_edge_index()
    with pytest.raises(ValueError):
        build_evidence_subgraph(edge_index, np.array([0.1, 0.2]), target_idx=0)


def test_top_features_sorted_and_filtered():
    edge_index = _star_edge_index()
    edge_mask = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    feat_mask = np.zeros((7, 4))
    feat_mask[0] = [0.1, 0.9, 0.0, 0.4]
    ev = build_evidence_subgraph(
        edge_index, edge_mask, target_idx=0, node_feature_mask=feat_mask, top_k_features=2
    )
    assert ev.top_features == [(1, 0.9), (3, 0.4)]
