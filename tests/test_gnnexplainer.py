import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from src._stubs.dummy_model import DummyModel
from src._stubs.ring import Ring
from src.explain.gnnexplainer import GNNExplainerWrapper


def _graph_with_one_dominant_edge():
    """target=0, connected to node 1 (large-magnitude features -> dominates SAGEConv's
    mean-aggregated message) and node 2 (near-zero features -> contributes almost nothing).
    A model that's actually graph-aware should score edge (0,1) as far more important than
    edge (0,2); a model that ignores its input (the old `torch.rand(...)` stub) would not.
    """
    x = torch.tensor([
        [0.1, 0.1, 0.1, 0.1],
        [5.0, 5.0, 5.0, 5.0],
        [0.001, 0.001, 0.001, 0.001],
    ])
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    return Data(x=x, edge_index=edge_index)


def test_removing_dominant_edge_changes_the_prediction():
    data = _graph_with_one_dominant_edge()
    model = DummyModel(in_dim=4, seed=0)
    model.eval()

    with torch.no_grad():
        full_score = model(data)["fraud_score"][0].item()

    keep = [
        i for i in range(data.edge_index.size(1))
        if 1 not in (int(data.edge_index[0, i]), int(data.edge_index[1, i]))
    ]
    reduced = Data(x=data.x, edge_index=data.edge_index[:, keep])
    with torch.no_grad():
        reduced_score = model(reduced)["fraud_score"][0].item()

    assert abs(full_score - reduced_score) > 1e-4


def test_gnnexplainer_assigns_higher_importance_to_the_dominant_edge():
    data = _graph_with_one_dominant_edge()
    model = DummyModel(in_dim=4, seed=0)
    model.eval()

    wrapper = GNNExplainerWrapper(model, config={"epochs": 150, "lr": 0.05, "seed": 0})
    result = wrapper.explain(data, target_idx=0)

    edge_index = data.edge_index.numpy()
    dominant_edge_ids = [
        i for i in range(edge_index.shape[1]) if 1 in (edge_index[0, i], edge_index[1, i])
    ]
    inert_edge_ids = [
        i for i in range(edge_index.shape[1]) if 2 in (edge_index[0, i], edge_index[1, i])
    ]

    dominant_weight = max(result.edge_mask[i] for i in dominant_edge_ids)
    inert_weight = max(result.edge_mask[i] for i in inert_edge_ids)

    assert dominant_weight > inert_weight


def test_repeated_explain_calls_are_deterministic():
    data = _graph_with_one_dominant_edge()
    model = DummyModel(in_dim=4, seed=0)
    model.eval()

    wrapper = GNNExplainerWrapper(model, config={"epochs": 50, "lr": 0.05, "seed": 7})

    result1 = wrapper.explain(data, target_idx=0)
    result2 = wrapper.explain(data, target_idx=0)

    assert (result1.edge_mask == result2.edge_mask).all()
    assert (result1.node_feature_mask == result2.node_feature_mask).all()
    assert result1.prediction == result2.prediction


def test_explain_ring_never_targets_an_entity_node():
    # star graph: node 0 is the entity hub (label -1), nodes 1-3 are scoreable accounts
    x = torch.randn(4, 4)
    src = [1, 2, 3, 0, 0, 0]
    dst = [0, 0, 0, 1, 2, 3]
    edge_index = torch.tensor([src, dst])
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor([-1, 1, 1, 1])

    model = DummyModel(in_dim=4, seed=0)
    model.eval()
    wrapper = GNNExplainerWrapper(model, config={"epochs": 30, "lr": 0.05, "seed": 0})

    ring = Ring(ring_id="r_test", node_ids=[0, 1, 2, 3], edge_list=list(zip(src, dst)),
                score=0.9, size=3, pattern="star")
    result = wrapper.explain_ring(data, ring)

    assert result.target_idx != 0


def test_explain_ring_raises_when_every_member_is_an_entity_node():
    x = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor([-1, -1])

    model = DummyModel(in_dim=4, seed=0)
    model.eval()
    wrapper = GNNExplainerWrapper(model, config={"epochs": 10, "lr": 0.05, "seed": 0})

    ring = Ring(ring_id="r_bad", node_ids=[0, 1], edge_list=[(0, 1)], score=0.5, size=2,
                pattern="unknown")

    with pytest.raises(ValueError, match="no scoreable members"):
        wrapper.explain_ring(data, ring)


def test_neighbourhood_path_matches_shapes_and_prediction():
    """On a graph above max_full_graph_nodes, explain() uses the target's k-hop neighbourhood.
    Masks must still be full-graph shaped, zero outside the neighbourhood, and the reported
    prediction must equal the model's own score for the target."""
    torch.manual_seed(0)
    n = 40
    x = torch.randn(n, 4)
    # two disconnected pieces: a chain 0-1-2-3-4 and a separate chain 20-21-22
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (20, 21), (21, 22)]
    src = [a for a, b in pairs] + [b for a, b in pairs]
    dst = [b for a, b in pairs] + [a for a, b in pairs]
    data = Data(x=x, edge_index=torch.tensor([src, dst]))

    from src.models.graphguard import GraphGuardModel
    model = GraphGuardModel(in_dim=4, hidden_dim=8, num_sage_layers=2, num_gat_layers=1,
                            gat_heads=2, dropout=0.0, latent_dim=4)
    model.eval()
    wrapper = GNNExplainerWrapper(model, config={"epochs": 5, "max_full_graph_nodes": 10})
    assert wrapper.num_hops == 3
    out = wrapper.explain(data, target_idx=0)

    assert out.edge_mask.shape == (data.edge_index.size(1),)
    assert out.node_feature_mask.shape[0] == n
    # edges of the far-away chain (20-21-22) can't influence node 0
    far_cols = [i for i in range(data.edge_index.size(1)) if int(data.edge_index[0, i]) >= 20]
    assert all(out.edge_mask[i] == 0 for i in far_cols)
    assert (out.node_feature_mask[20:] == 0).all()
    with torch.no_grad():
        expected = float(model(data)["fraud_score"][0])
    assert abs(out.prediction - expected) < 1e-5
