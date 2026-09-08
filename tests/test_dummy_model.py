"""Tests for DummyModel. Runs on a tiny synthetic graph -- no dataset download needed.

Per CONTRACTS.md: assert output shapes and value ranges, not accuracy (there's no accuracy
to test -- the outputs are random by design).
"""
import torch
from torch_geometric.data import Data

from src.models.dummy import DummyModel


def make_synthetic_graph(n_nodes: int = 10, n_edges: int = 20) -> Data:
    """A small random graph shaped like what to_pyg() would hand B: node features,
    an edge_index, and num_nodes -- nothing dataset-specific."""
    x = torch.randn(n_nodes, 16)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    return Data(x=x, edge_index=edge_index, num_nodes=n_nodes)


def test_output_keys_present():
    model = DummyModel(hidden_dim=32)
    data = make_synthetic_graph()
    out = model(data)
    assert set(out.keys()) == {"fraud_score", "anomaly_score", "embeddings", "attention"}


def test_output_shapes():
    n_nodes, n_edges, hidden_dim = 10, 20, 32
    model = DummyModel(hidden_dim=hidden_dim)
    data = make_synthetic_graph(n_nodes, n_edges)
    out = model(data)

    assert out["fraud_score"].shape == (n_nodes,)
    assert out["anomaly_score"].shape == (n_nodes,)
    assert out["embeddings"].shape == (n_nodes, hidden_dim)
    if out["attention"] is not None:
        assert out["attention"].shape == (n_edges,)


def test_fraud_score_in_unit_range():
    model = DummyModel()
    data = make_synthetic_graph()
    out = model(data)
    assert torch.all(out["fraud_score"] >= 0.0)
    assert torch.all(out["fraud_score"] <= 1.0)


def test_anomaly_score_non_negative():
    model = DummyModel()
    data = make_synthetic_graph()
    out = model(data)
    assert torch.all(out["anomaly_score"] >= 0.0)


def test_attention_can_be_none_or_correct_shape():
    """Attention is sometimes None by design -- consumers (C) must handle both branches.
    Force always_return_attention to check the non-None shape deterministically."""
    n_nodes, n_edges = 10, 20
    model = DummyModel(always_return_attention=True)
    data = make_synthetic_graph(n_nodes, n_edges)
    out = model(data)
    assert out["attention"] is not None
    assert out["attention"].shape == (n_edges,)


def test_load_returns_dummy_model():
    model = DummyModel.load("nonexistent/checkpoint.pt")
    assert isinstance(model, DummyModel)


def test_zero_edges_does_not_crash():
    """Edge case C might hit early: an isolated node with no edges yet."""
    data = make_synthetic_graph(n_nodes=5, n_edges=0)
    model = DummyModel()
    out = model(data)
    assert out["fraud_score"].shape == (5,)
    assert out["attention"] is None or out["attention"].shape == (0,)
