"""GraphBundle -> PyTorch Geometric objects, plus network features.

Kept separate from the loaders so M1 can work without torch installed and
M2 gets a stable handoff point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base import GraphBundle
from src.utils.config import get_logger

log = get_logger(__name__)


def add_network_features(bundle: GraphBundle, node_type: str | None = None) -> GraphBundle:
    """Append degree / clustering / pagerank to a node type's features.

    These are the 'social network metrics' Sheu & Li (2022) used as node
    features, so this is a direct reimplementation of their input design.
    """
    import networkx as nx

    nt = node_type or bundle.target_node_type
    rel = next((r for r in bundle.edges if r[0] == nt and r[2] == nt), None)
    if rel is None:
        log.warning("no %s-%s edges; skipping network features", nt, nt)
        return bundle

    n = len(bundle.nodes[nt])
    e = bundle.edges[rel]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(e["src"].to_numpy(), e["dst"].to_numpy()))

    log.info("computing network features on %d nodes / %d edges", n, G.number_of_edges())
    deg = np.array([d for _, d in sorted(G.degree())], dtype=np.float32)
    clus = np.array([v for _, v in sorted(nx.clustering(G).items())], dtype=np.float32)
    pr = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-4)
    pra = np.array([pr[i] for i in range(n)], dtype=np.float32)

    extra = np.stack([np.log1p(deg), clus, pra * n], axis=1).astype(np.float32)
    bundle.features[nt] = np.concatenate([bundle.features[nt], extra], axis=1)
    log.info("added 3 network features -> %d total", bundle.features[nt].shape[1])
    return bundle


def normalize_features(bundle: GraphBundle, train_mask: np.ndarray | None = None) -> GraphBundle:
    """Z-score each node type. Statistics come from the TRAIN rows only —
    using the full set would leak test information into the scaler."""
    for nt, X in bundle.features.items():
        m = train_mask if (train_mask is not None and nt == bundle.target_node_type) \
            else np.ones(len(X), dtype=bool)
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd == 0] = 1.0
        bundle.features[nt] = ((X - mu) / sd).astype(np.float32)
    return bundle


def to_pyg(bundle: GraphBundle, hetero: bool | None = None):
    """Convert to torch_geometric Data (homogeneous) or HeteroData."""
    import torch
    from torch_geometric.data import Data, HeteroData

    if hetero is None:
        hetero = len(bundle.nodes) > 1

    if not hetero:
        nt = bundle.target_node_type
        rel = next(r for r in bundle.edges if r[0] == nt and r[2] == nt)
        e = bundle.edges[rel]
        ei = torch.tensor(np.stack([e["src"].to_numpy(), e["dst"].to_numpy()]), dtype=torch.long)
        ei = torch.cat([ei, ei.flip(0)], dim=1)          # make it undirected
        return Data(
            x=torch.tensor(bundle.features[nt]),
            edge_index=ei,
            y=torch.tensor(bundle.labels[nt], dtype=torch.long),
            time=torch.tensor(bundle.time[nt], dtype=torch.long),
        )

    data = HeteroData()
    for nt, X in bundle.features.items():
        data[nt].x = torch.tensor(X)
        if nt in bundle.labels:
            data[nt].y = torch.tensor(bundle.labels[nt], dtype=torch.long)
        if nt in bundle.time:
            data[nt].time = torch.tensor(bundle.time[nt], dtype=torch.long)
    for (s, r, d), e in bundle.edges.items():
        ei = torch.tensor(np.stack([e["src"].to_numpy(), e["dst"].to_numpy()]), dtype=torch.long)
        data[s, r, d].edge_index = ei
        data[d, f"rev_{r}", s].edge_index = ei.flip(0)   # reverse edges for message passing
    return data
