"""Quantitative explanation metrics for the paper (not just the demo): fidelity and sparsity.

Both are defined against a `predict_fn(edge_index) -> float` callable rather than the raw
torch model directly, so this module has zero hard dependency on torch/PyG and is trivial to
unit test with a plain python function standing in for the model. `src/api/main.py` is what
supplies the real `predict_fn` (a closure over the loaded model + node features).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

PredictFn = Callable[[np.ndarray], float]  # takes edge_index [2, n_edges] -> fraud_score at target


@dataclass
class FidelityResult:
    original_score: float
    evidence_only_score: float
    fidelity: float  # in [0, 1]; higher = evidence subgraph alone reproduces the prediction well


def fidelity(
    predict_fn: PredictFn,
    full_edge_index: np.ndarray,
    evidence_edges: Sequence[Tuple[int, int]],
) -> FidelityResult:
    """Does keeping *only* the evidence subgraph reproduce the original prediction?

    We define fidelity as `1 - |original_score - evidence_only_score|`, i.e. 1.0 means the
    evidence subgraph alone gives an identical prediction to the full graph, 0.0 means it gives
    the opposite. This is the standard "fidelity-" definition (fidelity to a *sufficient*
    explanation) used in the GNNExplainer literature -- it directly answers "is this
    explanation faithful, or did we just show a plausible-looking but irrelevant subgraph?".

    Args:
        predict_fn: callable that takes an edge_index array [2, n_edges] and returns the
            model's fraud_score for the target node under that graph. Callers pass a closure
            that fixes the model, node features, and target node.
        full_edge_index: [2, n_edges] the complete graph's edges.
        evidence_edges: the (src, dst) pairs kept in the evidence subgraph.

    Returns:
        FidelityResult with both raw scores and the fidelity value.
    """
    original_score = predict_fn(np.asarray(full_edge_index))

    if not evidence_edges:
        evidence_edge_index = np.empty((2, 0), dtype=np.asarray(full_edge_index).dtype)
    else:
        evidence_edge_index = np.array(evidence_edges, dtype=np.asarray(full_edge_index).dtype).T
    evidence_only_score = predict_fn(evidence_edge_index)

    fid = 1.0 - abs(original_score - evidence_only_score)
    fid = max(0.0, min(1.0, fid))
    return FidelityResult(float(original_score), float(evidence_only_score), float(fid))


def sparsity(evidence_edge_count: int, total_edge_count: int) -> float:
    """Fraction of the graph's edges the evidence subgraph does *not* use.

    1.0 = maximally sparse (evidence uses almost no edges, very readable). 0.0 = evidence is
    the whole graph (not really an explanation). We report `1 - kept/total` rather than the
    raw kept-edge count so it's comparable across rings/nodes of very different neighbourhood
    sizes -- a fixed `top_k_edges` cap already bounds absolute size, sparsity is what tells you
    whether that cap actually mattered for this particular case.
    """
    if total_edge_count <= 0:
        raise ValueError("total_edge_count must be positive")
    if evidence_edge_count > total_edge_count:
        raise ValueError("evidence_edge_count cannot exceed total_edge_count")
    return 1.0 - (evidence_edge_count / total_edge_count)
