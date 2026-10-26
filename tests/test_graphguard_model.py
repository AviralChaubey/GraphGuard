"""Tests for GraphGuardModel. Tiny synthetic graph, no dataset download -- shapes and
ranges only, per CONTRACTS.md (there's no "correct" prediction to test against here)."""
import torch
from torch_geometric.data import Data, HeteroData

from src.models.graphguard import GraphGuardModel


def make_synthetic_graph(n_nodes: int = 20, n_edges: int = 60, n_feat: int = 16) -> Data:
    x = torch.randn(n_nodes, n_feat)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    return Data(x=x, edge_index=edge_index, num_nodes=n_nodes)


def make_synthetic_hetero_graph(n_tx: int = 20, n_card: int = 5, n_edges: int = 15,
                                 tx_feat: int = 8, card_feat: int = 3) -> HeteroData:
    """Mirrors IEEE-CIS's shape: one labelled target type (transaction) plus one
    unlabelled entity type (card), connected both ways (to_pyg() always adds the
    reverse relation too)."""
    data = HeteroData()
    data["transaction"].x = torch.randn(n_tx, tx_feat)
    data["card"].x = torch.randn(n_card, card_feat)
    ei = torch.stack([torch.randint(0, n_tx, (n_edges,)), torch.randint(0, n_card, (n_edges,))])
    data["transaction", "used_card", "card"].edge_index = ei
    data["card", "rev_used_card", "transaction"].edge_index = ei.flip(0)
    data.target_node_type = "transaction"
    return data


def make_hetero_model(in_dims=None, hidden_dim: int = 8) -> GraphGuardModel:
    in_dims = in_dims or {"transaction": 8, "card": 3}
    return GraphGuardModel(
        hidden_dim=hidden_dim, num_sage_layers=2, num_gat_layers=1,
        gat_heads=2, dropout=0.1, latent_dim=4,
        node_types=["transaction", "card"],
        edge_types=[("transaction", "used_card", "card"), ("card", "rev_used_card", "transaction")],
        target_node_type="transaction", in_dims=in_dims,
    )


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


# ---- Heterogeneous (IEEE-CIS-shaped) graph -------------------------------------
# These exercise the HeteroConv path added to replace the "homogeneous only for
# now" TODO -- same Contract 2 output shape, attention always None (see the
# module docstring for why), scored only for the target node type (transaction).

def test_hetero_forward_output_keys_and_shapes():
    n_tx, hidden_dim = 20, 8
    model = make_hetero_model(hidden_dim=hidden_dim)
    data = make_synthetic_hetero_graph(n_tx=n_tx)
    out = model(data)
    assert set(out.keys()) == {"fraud_score", "anomaly_score", "embeddings", "attention"}
    assert out["fraud_score"].shape == (n_tx,)
    assert out["anomaly_score"].shape == (n_tx,)
    assert out["embeddings"].shape == (n_tx, hidden_dim)
    assert out["attention"] is None  # see module docstring: not extracted in the hetero path


def test_hetero_fraud_score_in_unit_range():
    model = make_hetero_model()
    data = make_synthetic_hetero_graph()
    out = model(data)
    assert torch.all(out["fraud_score"] >= 0.0)
    assert torch.all(out["fraud_score"] <= 1.0)


def test_hetero_gat_dim_propagation_multi_layer():
    """num_gat_layers > 1 with heads > 1: the first GAT layer concatenates heads
    (output width hidden_dim * heads), so the next GAT layer's input width must
    follow that, not just assume hidden_dim -- this caught a real bug where the
    hetero path hardcoded every GAT layer's input as hidden_dim regardless of the
    previous layer's concat/heads."""
    data = make_synthetic_hetero_graph()
    model = GraphGuardModel(
        hidden_dim=8, num_sage_layers=1, num_gat_layers=2, gat_heads=2,
        dropout=0.1, latent_dim=4,
        node_types=["transaction", "card"],
        edge_types=[("transaction", "used_card", "card"), ("card", "rev_used_card", "transaction")],
        target_node_type="transaction", in_dims={"transaction": 8, "card": 3},
    )
    out = model(data)
    assert out["fraud_score"].shape == (20,)


def test_hetero_gradients_flow():
    # Only the target node type (transaction) feeds fraud_head/decoder, so the
    # LAST GAT layer's conv for any edge type whose destination is NOT the
    # target type produces an embedding nobody reads back -- that direction's
    # weights legitimately get no gradient, at any depth (it's not something
    # adding more layers fixes, since it's always the final layer's own output
    # that's unused). That's expected, not a bug -- so this test excludes
    # exactly that one dead-end conv and asserts every other parameter
    # (including earlier-layer convs for the same edge type) gets a gradient.
    model = GraphGuardModel(
        hidden_dim=8, num_sage_layers=1, num_gat_layers=2, gat_heads=2,
        dropout=0.1, latent_dim=4,
        node_types=["transaction", "card"],
        edge_types=[("transaction", "used_card", "card"), ("card", "rev_used_card", "transaction")],
        target_node_type="transaction", in_dims={"transaction": 8, "card": 3},
    )
    data = make_synthetic_hetero_graph()
    out = model.forward_with_logits(data)
    loss = out["fraud_logits"].sum() + out["reconstruction"].sum()
    loss.backward()

    last_gat_idx = len(model.gat_layers) - 1
    dead_end_prefix = f"gat_layers.{last_gat_idx}.convs.<transaction___used_card___card>"
    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is None and not n.startswith(dead_end_prefix)
    ]
    assert missing == []


def test_hetero_save_and_load_roundtrip(tmp_path):
    model = make_hetero_model()
    model.eval()
    ckpt_path = tmp_path / "hetero_test.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": dict(
                hidden_dim=8, num_sage_layers=2, num_gat_layers=1, gat_heads=2,
                dropout=0.1, latent_dim=4,
                node_types=["transaction", "card"],
                edge_types=[("transaction", "used_card", "card"),
                            ("card", "rev_used_card", "transaction")],
                target_node_type="transaction",
                in_dims={"transaction": 8, "card": 3},
            ),
        },
        ckpt_path,
    )
    loaded = GraphGuardModel.load(str(ckpt_path))
    loaded.eval()
    data = make_synthetic_hetero_graph()
    with torch.no_grad():
        out_original = model(data)
        out_loaded = loaded(data)
    assert torch.allclose(out_original["fraud_score"], out_loaded["fraud_score"])
