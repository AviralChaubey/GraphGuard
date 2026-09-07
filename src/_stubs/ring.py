"""Placeholder for B's `src/ring/extract.py` Ring dataclass, plus a synthetic-graph builder
used only by my tests. Delete once B's real `src/ring/extract.py` lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch_geometric.data import Data


@dataclass
class Ring:
    ring_id: str
    node_ids: List[int]
    edge_list: List[Tuple[int, int]]
    score: float
    size: int
    pattern: str = "unknown"  # 'star' | 'chain' | 'cyclic' | 'dense' | 'unknown'


def synthetic_star_graph(n_leaves: int = 6, n_feat: int = 8, seed: int = 0) -> Tuple[Data, Ring]:
    """A small star-shaped ring (one hub device, several cards) for tests and demos.

    Mirrors the IEEE-CIS style described in docs/team/PROMPT_C_explain_api_ui.md: node 0 is a
    'device' entity (label -1, never scored), nodes 1..n are 'card' accounts.
    """
    g = torch.Generator().manual_seed(seed)
    n_nodes = n_leaves + 1
    x = torch.randn(n_nodes, n_feat, generator=g)

    src = list(range(1, n_nodes)) + [0] * n_leaves
    dst = [0] * n_leaves + list(range(1, n_nodes))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # A per-edge scalar (e.g. "relation strength") -- present here purely to exercise the
    # edge_attr passthrough plumbing (_ModelAdapter -> Explainer -> model.forward) end to
    # end. DummyModel's SAGEConv layers don't consume it, but a real GATConv-based model
    # might, and that path needs to work without any changes to src/explain/ or src/api/.
    edge_attr = torch.rand(edge_index.size(1), 1, generator=g)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_types = ["device"] + ["card"] * n_leaves
    data.node_ids = ["device:D-3391"] + [f"card:{1000 + i}|404" for i in range(n_leaves)]
    labels = torch.tensor([-1] + [1] * n_leaves)
    data.labels = labels
    data.y = labels  # idiomatic PyG attribute name; see gnnexplainer.get_node_labels()

    ring = Ring(
        ring_id="r_00042",
        node_ids=list(range(n_nodes)),
        edge_list=list(zip(src, dst)),
        score=0.91,
        size=n_leaves,
        pattern="star",
    )
    return data, ring
