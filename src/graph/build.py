"""GraphBundle -> PyTorch Geometric objects, plus network features.

Kept separate from the loaders so M1 can work without torch installed and
M2 gets a stable handoff point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base import GraphBundle, temporal_split
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


def to_pyg(bundle: GraphBundle, hetero: bool | None = None,
           train_mask=None, test_mask=None, train_ratio: float = 0.7):
    """Convert to torch_geometric Data (homogeneous) or HeteroData.

    Always attaches `train_mask` and `test_mask` to the target node type, so a
    consumer can never accidentally train on the whole graph.

    If masks are not supplied they are computed with a TEMPORAL split (train on
    earlier time-steps, test on later ones) — never random. Prefer passing the
    masks saved in data/processed/<dataset>/splits.npz so every experiment runs
    on byte-identical partitions; `load_pyg()` below does that for you.

    Note for heterogeneous datasets (IEEE-CIS): this returns a **HeteroData**,
    not a Data. Masks land on data[target_node_type]. Entity node types
    (card/device/address/email) carry y = -1 and are never in either mask —
    they are conduits for message passing, not things to classify.
    """
    import torch
    from torch_geometric.data import Data, HeteroData

    if hetero is None:
        hetero = len(bundle.nodes) > 1

    nt = bundle.target_node_type
    t = bundle.time[nt]

    if train_mask is None or test_mask is None:
        log.warning("no split supplied — computing a temporal split "
                    "(train_ratio=%.2f). Pass splits.npz for reproducibility.",
                    train_ratio)
        train_mask, test_mask = temporal_split(t, train_ratio)

    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    if len(train_mask) != len(bundle.labels[nt]):
        raise ValueError(
            f"mask length {len(train_mask)} != {len(bundle.labels[nt])} nodes "
            f"for '{nt}' — the split does not belong to this bundle")
    if (train_mask & test_mask).any():
        raise ValueError("train and test masks overlap")

    # Unknown labels (-1) must never enter the supervised loss. Excluding them
    # here means a consumer cannot forget to.
    labelled = bundle.labels[nt] >= 0
    tr = torch.tensor(train_mask & labelled)
    te = torch.tensor(test_mask & labelled)

    if not hetero:
        rel = next(r for r in bundle.edges if r[0] == nt and r[2] == nt)
        e = bundle.edges[rel]
        ei = torch.tensor(np.stack([e["src"].to_numpy(), e["dst"].to_numpy()]),
                          dtype=torch.long)
        ei = torch.cat([ei, ei.flip(0)], dim=1)          # make it undirected
        return Data(
            x=torch.tensor(bundle.features[nt]),
            edge_index=ei,
            y=torch.tensor(bundle.labels[nt], dtype=torch.long),
            time=torch.tensor(t, dtype=torch.long),
            train_mask=tr,
            test_mask=te,
            target_node_type=nt,
        )

    data = HeteroData()
    for ntype, X in bundle.features.items():
        data[ntype].x = torch.tensor(X)
        if ntype in bundle.labels:
            data[ntype].y = torch.tensor(bundle.labels[ntype], dtype=torch.long)
        if ntype in bundle.time:
            data[ntype].time = torch.tensor(bundle.time[ntype], dtype=torch.long)
    data[nt].train_mask = tr
    data[nt].test_mask = te
    data.target_node_type = nt

    for (s_t, r, d_t), e in bundle.edges.items():
        ei = torch.tensor(np.stack([e["src"].to_numpy(), e["dst"].to_numpy()]),
                          dtype=torch.long)
        data[s_t, r, d_t].edge_index = ei
        data[d_t, f"rev_{r}", s_t].edge_index = ei.flip(0)   # bidirectional passing
    return data


def load_pyg(dataset: str, train_ratio: float = 0.7, **loader_kwargs):
    """One call: load a dataset, apply its saved split, return a PyG object.

    This is the recommended entry point for training and evaluation — it uses
    the split saved by scripts/preprocess.py, so every experiment runs on
    identical partitions.

        from src.graph import load_pyg
        data = load_pyg("elliptic")
    """
    from src.data import load_dataset
    from src.utils.config import PROJECT_ROOT

    bundle = load_dataset(dataset, **loader_kwargs)
    f = PROJECT_ROOT / "data" / "processed" / dataset / "splits.npz"
    if f.exists():
        z = np.load(f)
        log.info("using saved split from %s", f)
        return to_pyg(bundle, train_mask=z["train"], test_mask=z["test"])
    log.warning("%s not found — run scripts/preprocess.py --dataset %s first; "
                "falling back to a fresh temporal split", f, dataset)
    return to_pyg(bundle, train_ratio=train_ratio)
