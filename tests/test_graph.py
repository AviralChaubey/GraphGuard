import numpy as np

from src.data.base import temporal_split
from src.graph.build import normalize_features
from src.graph.sampler import NeighbourhoodSampler


def test_sampler_grows_with_hops(hetero_bundle):
    s = NeighbourhoodSampler(hetero_bundle)
    assert s.sample([0], k=1).n_nodes <= s.sample([0], k=2).n_nodes


def test_sampler_cap_limits_size(hetero_bundle):
    """Without a cap, one hop through a hub entity pulls in the whole graph."""
    s = NeighbourhoodSampler(hetero_bundle)
    big = s.sample([0, 1, 2], k=2, max_per_hop=10_000)
    small = s.sample([0, 1, 2], k=2, max_per_hop=3)
    assert small.n_nodes < big.n_nodes


def test_sampler_is_deterministic(hetero_bundle):
    s = NeighbourhoodSampler(hetero_bundle)
    a = s.sample([0, 5], k=2, max_per_hop=5, rng_seed=1)
    b = s.sample([0, 5], k=2, max_per_hop=5, rng_seed=1)
    assert a.n_nodes == b.n_nodes


def test_sampler_local_edges_are_in_range(hetero_bundle):
    s = NeighbourhoodSampler(hetero_bundle)
    sub = s.sample([0, 1], k=2)
    for (src_t, _, dst_t), e in sub.edges.items():
        assert e["src"].max() < len(sub.node_types[src_t])
        assert e["dst"].max() < len(sub.node_types[dst_t])


def test_normalize_uses_train_stats_only(toy_bundle):
    """Fitting the scaler on everything leaks test statistics into training."""
    tr, _ = temporal_split(toy_bundle.time["account"], 0.7)
    normalize_features(toy_bundle, tr)
    X = toy_bundle.features["account"]
    assert abs(X[tr].mean()) < 1e-5          # train is centred
    assert abs(X.mean()) > 1e-9              # the full set is not


# --- to_pyg contract with workstream B -------------------------------------

def _torch_available():
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


def test_to_pyg_attaches_masks(toy_bundle):
    """B consumes data.train_mask / data.test_mask — they must always exist."""
    if not _torch_available():
        return
    from src.graph.build import to_pyg
    d = to_pyg(toy_bundle)
    assert hasattr(d, "train_mask") and hasattr(d, "test_mask")
    assert d.train_mask.sum() > 0 and d.test_mask.sum() > 0


def test_to_pyg_masks_exclude_unknown_labels(toy_bundle):
    """-1 must never enter the supervised loss."""
    if not _torch_available():
        return
    from src.graph.build import to_pyg
    d = to_pyg(toy_bundle)
    y = d.y.numpy()
    assert (y[d.train_mask.numpy()] >= 0).all()
    assert (y[d.test_mask.numpy()] >= 0).all()


def test_to_pyg_masks_are_disjoint_and_temporal(toy_bundle):
    if not _torch_available():
        return
    from src.graph.build import to_pyg
    d = to_pyg(toy_bundle)
    tr, te = d.train_mask.numpy(), d.test_mask.numpy()
    assert not (tr & te).any()
    t = d.time.numpy()
    assert t[tr].max() <= t[te].min()      # train is the past, test the future


def test_to_pyg_rejects_mismatched_mask(toy_bundle):
    if not _torch_available():
        return
    import numpy as np
    import pytest
    from src.graph.build import to_pyg
    bad = np.ones(5, dtype=bool)
    with pytest.raises(ValueError, match="does not belong to this bundle"):
        to_pyg(toy_bundle, train_mask=bad, test_mask=bad)
