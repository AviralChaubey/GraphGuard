from src.explain.evidence import EvidenceSubgraph
from src.explain.reason import ReasonContext, generate_reason


def _evidence(n_edges=3):
    return EvidenceSubgraph(
        target_idx=1,
        nodes=[0, 1, 2, 3],
        edges=[(1, 0), (2, 0), (3, 0)][:n_edges],
        edge_weights=[0.9, 0.8, 0.7][:n_edges],
        top_features=[(4, 0.41)],
    )


def test_shared_entity_with_inflow_is_deterministic_and_matches_template():
    ctx = ReasonContext(
        node_id_map={0: "device:D-3391"},
        shared_entity_idx=0,
        shared_entity_count=6,
        inflow_pct=82,
        inflow_window_hours=24,
    )
    ev = _evidence()
    r1 = generate_reason(ev, ctx)
    r2 = generate_reason(ev, ctx)  # same inputs -> must be byte-identical, no randomness

    assert r1 == r2
    assert "6 accounts share device:D-3391" in r1
    assert "82% of inflow in 24h" in r1


def test_shared_entity_without_inflow_falls_back_gracefully():
    ctx = ReasonContext(
        node_id_map={0: "device:D-3391"}, shared_entity_idx=0, shared_entity_count=6,
    )
    reason = generate_reason(_evidence(), ctx)
    assert "6 accounts share device:D-3391" in reason
    assert "inflow" not in reason


def test_anonymised_dataset_gets_structural_only_reason():
    ctx = ReasonContext(feature_names=None)  # e.g. Elliptic: no shared entity, no feature names
    reason = generate_reason(_evidence(n_edges=2), ctx)
    assert "structure" in reason.lower()
    assert "anonymised" in reason.lower()


def test_no_evidence_edges_says_so_explicitly():
    ev = EvidenceSubgraph(target_idx=1, nodes=[1], edges=[], edge_weights=[], top_features=[])
    ctx = ReasonContext()
    reason = generate_reason(ev, ctx)
    assert "no supporting edges" in reason.lower()


def test_known_feature_names_are_named_in_structural_reason():
    ctx = ReasonContext(feature_names=["fan_out", "f1", "f2", "f3", "amount_z"])
    ev = EvidenceSubgraph(
        target_idx=1, nodes=[0, 1], edges=[(1, 0)], edge_weights=[0.9],
        top_features=[(0, 0.41)],
    )
    reason = generate_reason(ev, ctx)
    assert "fan_out (0.41)" in reason
