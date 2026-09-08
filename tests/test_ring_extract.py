"""Tests for src/ring/extract.py. Builds small synthetic graphs with a known planted
topology (star / chain / cycle) and checks extract_rings() finds them correctly."""
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.ring.extract import Ring, _classify_pattern, extract_rings


def _to_pyg(g: nx.Graph, n_nodes: int) -> Data:
    edges = list(g.edges())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # undirected: add both directions, matching to_pyg()'s "always adds reverse edges" guarantee
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=n_nodes)


def test_classify_star():
    g = nx.star_graph(5)  # node 0 = hub, nodes 1-5 = leaves
    assert _classify_pattern(g) == "star"


def test_classify_cycle():
    g = nx.cycle_graph(6)
    assert _classify_pattern(g) == "cyclic"


def test_classify_chain():
    g = nx.path_graph(5)
    assert _classify_pattern(g) == "chain"


def test_classify_dense():
    g = nx.complete_graph(6)
    assert _classify_pattern(g) == "dense"


def test_classify_unknown_for_sparse_irregular_graph():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3), (1, 4)])  # irregular, not a clean shape
    assert _classify_pattern(g) in ("unknown", "dense")  # not asserting a specific label, just that it doesn't crash


def test_extract_rings_finds_planted_star():
    n_nodes = 20
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    # plant a star among nodes 0-5 (hub=0)
    for leaf in range(1, 6):
        g.add_edge(0, leaf)
    data = _to_pyg(g, n_nodes)

    scores = np.zeros(n_nodes)
    scores[0:6] = 0.9  # planted ring scores high
    scores[6:] = 0.1   # everyone else scores low

    rings = extract_rings(data, scores, threshold=0.7, min_size=3)
    assert len(rings) >= 1
    assert all(isinstance(r, Ring) for r in rings)
    top = rings[0]
    assert top.size >= 3
    assert set(top.node_ids) <= set(range(6))  # only planted-ring nodes, no noise


def test_extract_rings_below_min_size_returns_empty():
    n_nodes = 10
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edge(0, 1)
    data = _to_pyg(g, n_nodes)

    scores = np.zeros(n_nodes)
    scores[0:2] = 0.9  # only 2 nodes above threshold -- below min_size=3

    rings = extract_rings(data, scores, threshold=0.7, min_size=3)
    assert rings == []


def test_extract_rings_no_nodes_above_threshold():
    n_nodes = 10
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    data = _to_pyg(g, n_nodes)
    scores = np.full(n_nodes, 0.1)

    rings = extract_rings(data, scores, threshold=0.7, min_size=3)
    assert rings == []


def test_extract_rings_accepts_torch_tensor_scores():
    n_nodes = 10
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for leaf in range(1, 5):
        g.add_edge(0, leaf)
    data = _to_pyg(g, n_nodes)

    scores = torch.zeros(n_nodes)
    scores[0:5] = 0.9

    rings = extract_rings(data, scores, threshold=0.7, min_size=3)
    assert len(rings) >= 1


def test_ring_score_is_mean_of_member_scores():
    n_nodes = 10
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for leaf in range(1, 5):
        g.add_edge(0, leaf)
    data = _to_pyg(g, n_nodes)

    scores = np.zeros(n_nodes)
    scores[[0, 1, 2, 3, 4]] = [0.8, 0.9, 1.0, 0.7, 0.6]

    rings = extract_rings(data, scores, threshold=0.5, min_size=3)
    assert len(rings) == 1
    expected = np.mean([0.8, 0.9, 1.0, 0.7, 0.6])
    assert abs(rings[0].score - expected) < 1e-6
