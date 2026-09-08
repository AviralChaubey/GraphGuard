"""Ring extraction: threshold high-scoring nodes -> induced subgraph -> Louvain communities
-> classify each community's topology into Star / Chain / Cyclic / Dense / Unknown.

Contract (docs/CONTRACTS.md, Contract 2): extract_rings() returns list[Ring] with this exact
dataclass shape -- C's /ring/{ring_id} endpoint depends on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch

from src.utils.config import get_logger

logger = get_logger(__name__)


@dataclass
class Ring:
    ring_id: str
    node_ids: list[int]
    edge_list: list[tuple]
    score: float
    size: int
    pattern: str  # 'star' | 'chain' | 'cyclic' | 'dense' | 'unknown'


def _classify_pattern(g: nx.Graph) -> str:
    """Heuristic topology classifier, following the Star/Chain/Cyclic typology from
    Cheng et al. 2023 (GAGNN) plus a Dense catch-all for anything too tightly connected
    to fit those three shapes cleanly.

    Order matters: check Dense first (it can coincidentally satisfy loose degree checks),
    then the three named laundering topologies, then fall back to Unknown rather than
    force-fitting a pattern a human reviewer would disagree with.
    """
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n < 2:
        return "unknown"

    max_possible_edges = n * (n - 1) / 2
    density = m / max_possible_edges if max_possible_edges > 0 else 0.0

    degrees = sorted([d for _, d in g.degree()], reverse=True)

    # Dense: most possible edges present -- a tightly-knit clique-like cluster, not a
    # deliberate laundering shape.
    if density >= 0.7:
        return "dense"

    # Star: one hub connected to every other node, which are otherwise leaves (degree 1).
    if degrees[0] == n - 1 and all(d == 1 for d in degrees[1:]):
        return "star"

    # Cyclic: every node has degree exactly 2 and the graph is a single connected cycle
    # (n nodes, n edges, connected).
    if all(d == 2 for d in degrees) and m == n and nx.is_connected(g):
        return "cyclic"

    # Chain: a simple path -- two endpoints of degree 1, everyone else degree 2, n-1 edges,
    # connected.
    if (
        m == n - 1
        and nx.is_connected(g)
        and degrees.count(1) == 2
        and all(d in (1, 2) for d in degrees)
    ):
        return "chain"

    return "unknown"


def extract_rings(data, scores: torch.Tensor | np.ndarray, threshold: float = 0.7, min_size: int = 3) -> list["Ring"]:
    """Threshold nodes by `scores`, build the induced subgraph on the survivors, split it
    into Louvain communities, and classify each community's topology.

    Args:
        data: a torch_geometric Data (or anything exposing `.edge_index` and `.num_nodes`).
        scores: per-node risk score in [0, 1], e.g. GraphGuardModel's `fraud_score` output,
            or fraud_score and anomaly_score combined upstream (that combination is a modeling
            choice left to the caller, not baked in here).
        threshold: nodes with score >= threshold are candidates for ring membership.
        min_size: communities smaller than this are dropped (too small to call a "ring").

    Returns:
        One Ring per surviving Louvain community, sorted by score descending.
    """
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    candidate_nodes = np.where(scores_np >= threshold)[0]
    logger.info(
        "extract_rings: %d/%d nodes above threshold=%.2f", len(candidate_nodes), len(scores_np), threshold
    )

    if len(candidate_nodes) < min_size:
        return []

    # Build the full graph once, then take the induced subgraph on candidates -- cheaper
    # than filtering edges twice, and networkx's induced_subgraph handles the edge-list
    # intersection correctly.
    edge_index = data.edge_index.detach().cpu().numpy() if torch.is_tensor(data.edge_index) else data.edge_index
    full_graph = nx.Graph()
    full_graph.add_nodes_from(range(int(data.num_nodes)))
    full_graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    induced = full_graph.subgraph(candidate_nodes.tolist())

    communities = nx.algorithms.community.louvain_communities(induced, seed=42)

    rings: list[Ring] = []
    for i, community in enumerate(communities):
        if len(community) < min_size:
            continue
        node_ids = sorted(community)
        subgraph = induced.subgraph(node_ids)
        edge_list = list(subgraph.edges())
        ring_score = float(np.mean([scores_np[n] for n in node_ids]))
        pattern = _classify_pattern(subgraph)

        rings.append(
            Ring(
                ring_id=f"r_{i:05d}",
                node_ids=node_ids,
                edge_list=edge_list,
                score=ring_score,
                size=len(node_ids),
                pattern=pattern,
            )
        )

    rings.sort(key=lambda r: r.score, reverse=True)
    logger.info("extract_rings: %d rings found after min_size=%d filter", len(rings), min_size)
    return rings
