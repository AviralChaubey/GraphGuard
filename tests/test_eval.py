import numpy as np
import pytest

from src.data.base import temporal_split
from src.eval import compare, evaluate_detection, evaluate_rings, run_baselines


def test_unknown_labels_are_excluded():
    """-1 must never be scored. Counting unknowns as licit would reward a
    model for calling unexamined nodes clean."""
    y = np.array([1, 0, -1, -1, 1, 0])
    m = evaluate_detection(y, np.array([.9, .1, .8, .2, .7, .3]))
    assert m.n_evaluated == 4
    assert m.n_positive == 2


def test_perfect_and_random_scores_rank_correctly():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, .1, 800)
    perfect = evaluate_detection(y, y.astype(float) + rng.normal(0, .01, 800))
    noise = evaluate_detection(y, rng.random(800))
    assert perfect.pr_auc > 0.9
    assert perfect.pr_auc > noise.pr_auc


def test_raises_when_no_positives():
    with pytest.raises(ValueError, match="no labelled positives"):
        evaluate_detection(np.zeros(50, dtype=int), np.random.rand(50))


def test_precision_at_k_bounds():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, .2, 500)
    m = evaluate_detection(y, rng.random(500), ks=(10, 50, 100))
    assert set(m.precision_at_k) == {10, 50, 100}
    assert all(0 <= v <= 1 for v in m.precision_at_k.values())


def test_ring_metrics_pure_and_impure():
    y = np.array([1] * 10 + [0] * 10)
    pure = evaluate_rings([{"node_ids": list(range(10))}], y)
    assert pure.mean_purity == 1.0
    assert pure.coverage == 1.0
    mixed = evaluate_rings([{"node_ids": [0, 1, 10, 11]}], y)
    assert mixed.mean_purity == 0.5


def test_ring_metrics_accepts_dataclass():
    from dataclasses import dataclass

    @dataclass
    class R:
        node_ids: list

    y = np.array([1, 1, 0, -1])
    assert evaluate_rings([R([0, 1, 2])], y).n_rings == 1


def test_baselines_beat_majority(toy_bundle):
    tr, te = temporal_split(toy_bundle.time["account"], 0.7)
    res = run_baselines(toy_bundle, tr, te, include_network_features=False)
    assert "majority (never fraud)" in res
    best = max(v.pr_auc for k, v in res.items() if "majority" not in k)
    assert best > res["majority (never fraud)"].pr_auc


def test_compare_marks_the_leader(toy_bundle):
    tr, te = temporal_split(toy_bundle.time["account"], 0.7)
    txt = compare(run_baselines(toy_bundle, tr, te, include_network_features=False))
    assert "pr_auc" in txt and "%" in txt
