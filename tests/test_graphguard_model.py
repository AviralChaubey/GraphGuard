"""Tests for GraphGuardModel. Tiny synthetic graph, no dataset download -- shapes and
ranges only, per CONTRACTS.md (there's no "correct" prediction to test against here)."""
import torch
from torch_geometric.data import Data

from src.models.graphguard import GraphGuardModel


def make_synthetic_graph(n_nodes: int = 20, n_edges: int = 60, n_feat: int = 16) -> Data:
    x = torch.randn(n_nodes, n_feat)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    return Data(x=x, edge_index=edge_index, num_nodes=n_nodes)


def make_model(in_dim: int = 16, hidden_dim: int = 8) -> GraphGuardModel:
    # small dims so tests run fast
    return GraphGuardModel(
        in_dim=in_dim, hidden_dim=hidden_dim, num_sage_layers=2, num_gat_layers=2,
        gat_heads=2, dropout=0.1, latent_dim=4,
    )


def test_forward_output_keys():
    model = make_model()
    data = make_synthetic_graph()
    out = model(data)
    assert set(out.keys()) == {"fraud_score", "anomaly_score", "embeddings", "attention"}


def test_forward_output_shapes():
    n_nodes, hidden_dim = 20, 8
    model = make_model(hidden_dim=hidden_dim)
    data = make_synthetic_graph(n_nodes=n_nodes)
    out = model(data)

    assert out["fraud_score"].shape == (n_nodes,)
    assert out["anomaly_score"].shape == (n_nodes,)
    assert out["embeddings"].shape == (n_nodes, hidden_dim)
    assert out["attention"] is not None
    assert out["attention"].dim() == 1


def test_fraud_score_in_unit_range():
    model = make_model()
    data = make_synthetic_graph()
    out = model(data)
    assert torch.all(out["fraud_score"] >= 0.0)
    assert torch.all(out["fraud_score"] <= 1.0)


def test_anomaly_score_non_negative():
    model = make_model()
    data = make_synthetic_graph()
    out = model(data)
    assert torch.all(out["anomaly_score"] >= 0.0)


def test_forward_with_logits_matches_forward():
    """forward() and forward_with_logits() must agree on fraud_score / anomaly_score /
    embeddings -- forward_with_logits() only adds fields, it must never diverge in values."""
    model = make_model()
    model.eval()
    data = make_synthetic_graph()
    with torch.no_grad():
        out1 = model(data)
        torch.manual_seed(0)
        out2 = model.forward_with_logits(data)
    assert torch.allclose(out1["fraud_score"], torch.sigmoid(out2["fraud_logits"]))


def test_gradients_flow():
    """A cheap check that the whole stack is differentiable end to end -- catches a
    detached tensor or an accidental .item() silently breaking backprop."""
    model = make_model()
    data = make_synthetic_graph()
    out = model.forward_with_logits(data)
    loss = out["fraud_logits"].sum() + out["reconstruction"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_save_and_load_roundtrip(tmp_path):
    model = make_model()
    ckpt_path = tmp_path / "test.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": dict(
                in_dim=16, hidden_dim=8, num_sage_layers=2, num_gat_layers=2,
                gat_heads=2, dropout=0.1, latent_dim=4,
            ),
        },
        ckpt_path,
    )
    loaded = GraphGuardModel.load(str(ckpt_path))
    data = make_synthetic_graph()
    model.eval()
    loaded.eval()
    with torch.no_grad():
        out_original = model(data)
        out_loaded = loaded(data)
    assert torch.allclose(out_original["fraud_score"], out_loaded["fraud_score"])


def test_load_missing_checkpoint_raises():
    try:
        GraphGuardModel.load("does/not/exist.pt")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
