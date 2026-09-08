import numpy as np
import pytest

from src.explain.metrics import fidelity, sparsity


def test_fidelity_is_perfect_when_evidence_reproduces_prediction():
    # a fake model whose score only depends on whether edge (0,1) is present
    def predict_fn(edge_index: np.ndarray) -> float:
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        return 0.9 if (0, 1) in edges else 0.1

    full = np.array([[0, 2], [1, 3]])
    result = fidelity(predict_fn, full, evidence_edges=[(0, 1)])

    assert result.original_score == pytest.approx(0.9)
    assert result.evidence_only_score == pytest.approx(0.9)
    assert result.fidelity == pytest.approx(1.0)


def test_fidelity_penalises_evidence_that_changes_the_prediction():
    def predict_fn(edge_index: np.ndarray) -> float:
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        return 0.9 if (0, 1) in edges else 0.1

    full = np.array([[0, 2], [1, 3]])
    # evidence subgraph drops the edge that actually mattered -> big fidelity drop
    result = fidelity(predict_fn, full, evidence_edges=[(2, 3)])

    assert result.fidelity == pytest.approx(0.2)  # 1 - |0.9 - 0.1|


def test_fidelity_handles_empty_evidence():
    def predict_fn(edge_index: np.ndarray) -> float:
        return 0.5 if edge_index.shape[1] == 0 else 0.9

    full = np.array([[0], [1]])
    result = fidelity(predict_fn, full, evidence_edges=[])
    assert result.evidence_only_score == pytest.approx(0.5)


def test_sparsity_bounds_and_values():
    assert sparsity(0, 10) == 1.0
    assert sparsity(10, 10) == 0.0
    assert sparsity(3, 12) == pytest.approx(0.75)


def test_sparsity_rejects_invalid_input():
    with pytest.raises(ValueError):
        sparsity(5, 0)
    with pytest.raises(ValueError):
        sparsity(11, 10)
