"""k-hop neighbourhood sampling.

Why this exists: at inference time we must score a single new transaction
without materialising the whole graph. A GNN with L layers only ever sees L
hops, so the L-hop neighbourhood is provably sufficient — everything beyond it
is wasted computation.

Pure NumPy/pandas: no torch needed, so this runs in the API process too.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.base import GraphBundle
from src.utils.config import get_logger

log = get_logger(__name__)


@dataclass
class Subgraph:
    """A k-hop neighbourhood, re-indexed to 0..m-1.

    `node_map` maps local index -> original global index, which the
    explainability layer needs to name the nodes it points at.
    """
    node_types: dict          # local_type -> np.ndarray of ORIGINAL indices
    edges: dict               # (s, rel, d) -> DataFrame of LOCAL src/dst
    seed_local: dict          # node_type -> local index of each seed
    n_nodes: int

    def global_ids(self, node_type: str) -> np.ndarray:
        return self.node_types[node_type]


class NeighbourhoodSampler:
    """Builds an adjacency index once, then answers k-hop queries cheaply."""

    def __init__(self, bundle: GraphBundle):
        self.bundle = bundle
        self._adj: dict = {}
        for (s, rel, d), e in bundle.edges.items():
            src = e["src"].to_numpy(); dst = e["dst"].to_numpy()
            # both directions — message passing is bidirectional after to_pyg()
            self._adj[(s, rel, d)] = self._build(src, dst, len(bundle.nodes[s]))
            self._adj[(d, f"rev_{rel}", s)] = self._build(dst, src, len(bundle.nodes[d]))
        log.info("sampler ready: %d relations", len(self._adj))

    @staticmethod
    def _build(src: np.ndarray, dst: np.ndarray, n_src: int):
        """CSR-style index: sorted targets + offsets. O(1) lookup per node."""
        order = np.argsort(src, kind="stable")
        s_sorted, d_sorted = src[order], dst[order]
        counts = np.bincount(s_sorted, minlength=n_src)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        return d_sorted, offsets

    def _neighbours(self, rel_key, nodes: np.ndarray) -> np.ndarray:
        targets, offsets = self._adj[rel_key]
        if len(nodes) == 0:
            return np.empty(0, dtype=int)
        parts = [targets[offsets[n]:offsets[n + 1]] for n in nodes
                 if n + 1 < len(offsets)]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=int)

    def sample(self, seeds, node_type: str | None = None, k: int = 2,
               max_per_hop: int = 5000, rng_seed: int = 42) -> Subgraph:
        """Expand `seeds` outward k hops.

        max_per_hop caps each frontier. Without it, one hop through a hub entity
        (a shared 'Windows' device) pulls in the entire graph and the point of
        sampling is lost. Sampling is seeded, so an explanation is reproducible.
        """
        nt = node_type or self.bundle.target_node_type
        rng = np.random.default_rng(rng_seed)
        seeds = np.atleast_1d(np.asarray(seeds, dtype=int))

        reached: dict = {t: np.empty(0, dtype=int) for t in self.bundle.nodes}
        reached[nt] = seeds.copy()
        frontier = {nt: seeds.copy()}

        for hop in range(k):
            nxt: dict = {}
            for (s, rel, d) in self._adj:
                if s not in frontier or len(frontier[s]) == 0:
                    continue
                found = self._neighbours((s, rel, d), frontier[s])
                new = np.setdiff1d(found, reached[d], assume_unique=False)
                if len(new) > max_per_hop:
                    log.debug("hop %d %s->%s: capping %d to %d", hop, s, d, len(new), max_per_hop)
                    new = rng.choice(new, max_per_hop, replace=False)
                if len(new):
                    nxt[d] = np.union1d(nxt.get(d, np.empty(0, dtype=int)), new)
            for t, v in nxt.items():
                reached[t] = np.union1d(reached[t], v)
            frontier = nxt
            if not frontier:
                break

        # re-index to local 0..m-1 and keep only edges fully inside the subgraph
        local = {t: {g: i for i, g in enumerate(v)} for t, v in reached.items() if len(v)}
        edges = {}
        for (s, rel, d), e in self.bundle.edges.items():
            if s not in local or d not in local:
                continue
            ls, ld = local[s], local[d]
            m = e["src"].isin(ls.keys()) & e["dst"].isin(ld.keys())
            if not m.any():
                continue
            sub = e[m]
            edges[(s, rel, d)] = pd.DataFrame({
                "src": sub["src"].map(ls).to_numpy(),
                "dst": sub["dst"].map(ld).to_numpy()})

        return Subgraph(
            node_types={t: v for t, v in reached.items() if len(v)},
            edges=edges,
            seed_local={nt: np.array([local[nt][s] for s in seeds])},
            n_nodes=sum(len(v) for v in reached.values()),
        )
