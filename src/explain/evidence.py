"""Turns a raw GNNExplainer mask into a minimal *evidence subgraph* -- small enough that an
analyst can actually read it, per docs/team/PROMPT_C_explain_api_ui.md.

Deliberately has no dependency on torch/PyG: it operates on plain numpy arrays, so it's cheap
to unit test and can be reused if we ever swap the explainer algorithm out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.utils.config import get_logger, load_config

logger = get_logger(__name__)


@dataclass
class EvidenceSubgraph:
    target_idx: int
    nodes: List[int]                       # node indices touched by the kept edges (incl. target)
    edges: List[Tuple[int, int]]           # (src, dst) pairs, kept edges only
    edge_weights: List[float]              # mask weight per kept edge, same order as `edges`
    top_features: List[Tuple[int, float]]  # (feature_idx, importance), sorted descending
    edge_ids: List[int] = field(default_factory=list)  # original edge_index column indices,
    # same order as `edges` -- lets callers slice per-edge attributes (e.g. `data.edge_attr`)
    # for exactly the kept edges without having to re-match (src, dst) pairs, which would be
    # ambiguous if the graph ever has parallel edges between the same two nodes.


def build_evidence_subgraph(
    edge_index: np.ndarray,
    edge_mask: np.ndarray,
    target_idx: int,
    node_feature_mask: Optional[np.ndarray] = None,
    top_k_edges: Optional[int] = None,
    edge_threshold: Optional[float] = None,
    top_k_features: int = 5,
) -> EvidenceSubgraph:
    """Select the top-k highest-weight edges (above `edge_threshold`) touching the target
    node's local neighbourhood, and the feature importances for the target node.

    Args:
        edge_index: [2, n_edges] array of (src, dst) node indices.
        edge_mask: [n_edges] array of importance weights in [0, 1], aligned to `edge_index`
            columns (this is exactly what `GNNExplainerWrapper.explain` returns).
        target_idx: the node the explanation is for.
        node_feature_mask: optional [n_nodes, n_feat] or [n_feat] array of feature importances.
        top_k_edges: cap on number of edges kept. Defaults to `configs/default.yaml
            explain.top_k_edges` (falls back to 8) -- this is what keeps explanations
            human-readable rather than a dump of the whole neighbourhood.
        edge_threshold: drop edges below this weight even if inside top_k. Defaults to
            `configs/default.yaml explain.edge_threshold` (falls back to 0.1).
        top_k_features: how many top features to report.

    Returns:
        EvidenceSubgraph, ready for `src/explain/reason.py` and the `/explain/{id}` response.
    """
    cfg = load_config().get("explain", {})
    top_k_edges = top_k_edges if top_k_edges is not None else cfg.get("top_k_edges", 8)
    edge_threshold = edge_threshold if edge_threshold is not None else cfg.get("edge_threshold", 0.1)

    edge_index = np.asarray(edge_index)
    edge_mask = np.asarray(edge_mask)
    if edge_index.shape[1] != edge_mask.shape[0]:
        raise ValueError(
            f"edge_index has {edge_index.shape[1]} edges but edge_mask has {edge_mask.shape[0]}"
        )

    # Only consider edges touching the target's local neighbourhood -- an edge on the other
    # side of the graph might have a high global mask value but tells the analyst nothing
    # about *this* alert.
    touches_target = (edge_index[0] == target_idx) | (edge_index[1] == target_idx)
    candidate_edge_ids = np.where(touches_target)[0]

    if len(candidate_edge_ids) == 0:
        logger.warning("target_idx=%s has no incident edges; empty evidence subgraph", target_idx)
        return EvidenceSubgraph(target_idx, [target_idx], [], [], [], edge_ids=[])

    candidate_weights = edge_mask[candidate_edge_ids]
    order = np.argsort(-candidate_weights)  # descending
    kept = [
        candidate_edge_ids[i]
        for i in order[:top_k_edges]
        if candidate_weights[i] >= edge_threshold
    ]

    if not kept:
        # Nothing cleared the threshold -- keep the single strongest edge anyway so the
        # explanation is never completely empty; note this rather than silently returning
        # nothing, since an all-below-threshold ring is itself worth flagging to the analyst.
        best = candidate_edge_ids[order[0]]
        kept = [best]
        logger.info(
            "target_idx=%s: no edges cleared threshold=%.3f, keeping strongest (w=%.3f)",
            target_idx, edge_threshold, edge_mask[best],
        )

    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) for i in kept]
    edge_weights = [float(edge_mask[i]) for i in kept]
    edge_ids = [int(i) for i in kept]
    nodes = sorted({target_idx, *(n for pair in edges for n in pair)})

    top_features = _top_features(node_feature_mask, target_idx, top_k_features)

    return EvidenceSubgraph(target_idx, nodes, edges, edge_weights, top_features, edge_ids=edge_ids)


def _top_features(
    node_feature_mask: Optional[np.ndarray], target_idx: int, k: int
) -> List[Tuple[int, float]]:
    if node_feature_mask is None:
        return []
    mask = np.asarray(node_feature_mask)
    row = mask[target_idx] if mask.ndim == 2 else mask
    order = np.argsort(-row)[:k]
    return [(int(i), float(row[i])) for i in order if row[i] > 0]
