import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Data

from src._stubs.dummy_model import DummyModel


def _small_graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 6, generator=g)
    edge_index = torch.tensor([[0, 1, 2, 1, 3, 0], [1, 0, 1, 2, 0, 3]])
    return Data(x=x, edge_index=edge_index)


def test_forward_is_deterministic_across_repeated_calls():
    model = DummyModel(in_dim=6, seed=0)
    model.eval()
    data = _small_graph()

    with torch.no_grad():
        out1 = model(data)
        out2 = model(data)

    assert torch.equal(out1["fraud_score"], out2["fraud_score"])
    assert torch.equal(out1["anomaly_score"], out2["anomaly_score"])
    assert torch.equal(out1["embeddings"], out2["embeddings"])


def test_two_instances_with_same_seed_agree():
    data = _small_graph()
    m1 = DummyModel(in_dim=6, seed=0)
    m2 = DummyModel(in_dim=6, seed=0)
    m1.eval(); m2.eval()

    with torch.no_grad():
        assert torch.equal(m1(data)["fraud_score"], m2(data)["fraud_score"])


def test_output_depends_on_node_features():
    model = DummyModel(in_dim=6, seed=0)
    model.eval()
    data = _small_graph()
    perturbed = Data(x=data.x + 1.0, edge_index=data.edge_index)

    with torch.no_grad():
        original = model(data)["fraud_score"]
        changed = model(perturbed)["fraud_score"]

    assert not torch.allclose(original, changed)


def test_output_depends_on_graph_connectivity():
    model = DummyModel(in_dim=6, seed=0)
    model.eval()
    data = _small_graph()
    # drop the last edge pair (3,0)/(0,3)
    fewer_edges = Data(x=data.x, edge_index=data.edge_index[:, :4])

    with torch.no_grad():
        original = model(data)["fraud_score"]
        changed = model(fewer_edges)["fraud_score"]

    assert not torch.allclose(original, changed)


def test_shapes_and_ranges_match_contract_2():
    model = DummyModel(in_dim=6, seed=0)
    model.eval()
    data = _small_graph()

    with torch.no_grad():
        out = model(data)

    n, n_edges = data.x.size(0), data.edge_index.size(1)
    assert out["fraud_score"].shape == (n,)
    assert torch.all((out["fraud_score"] >= 0) & (out["fraud_score"] <= 1))
    assert out["anomaly_score"].shape == (n,)
    assert torch.all(out["anomaly_score"] >= 0)
    assert out["embeddings"].shape == (n, model.hidden_dim)
    assert out["attention"].shape == (n_edges,)


def test_gradients_flow_to_input_features():
    # GNNExplainer needs a non-trivial gradient landscape -- verify one exists at all.
    model = DummyModel(in_dim=6, seed=0)
    data = _small_graph()
    x = data.x.clone().requires_grad_(True)
    out = model(Data(x=x, edge_index=data.edge_index))
    out["fraud_score"][0].backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0)
