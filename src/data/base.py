"""Common structures shared by all dataset loaders.

Every loader returns a GraphBundle so the downstream model code never has to
care which dataset it came from. This is the contract between M1 and M2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GraphBundle:
    """Dataset-agnostic graph container.

    nodes      : node_type -> DataFrame indexed 0..n-1, must contain 'node_id'
    edges      : (src_type, relation, dst_type) -> DataFrame with 'src','dst' (integer indices)
    features   : node_type -> float32 array [n_nodes, n_features]
    labels     : node_type -> int array [n_nodes]; 1 fraud, 0 licit, -1 unknown
    time       : node_type -> int array [n_nodes]; time-step for temporal splitting
    """
    name: str
    nodes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    edges: Dict[tuple, pd.DataFrame] = field(default_factory=dict)
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    time: Dict[str, np.ndarray] = field(default_factory=dict)
    target_node_type: str = "transaction"

    def summary(self) -> str:
        lines = [f"GraphBundle<{self.name}>  target={self.target_node_type}"]
        for nt, df in self.nodes.items():
            feat = self.features.get(nt)
            lab = self.labels.get(nt)
            extra = f", feat={feat.shape[1]}" if feat is not None else ""
            if lab is not None:
                extra += (f", fraud={int((lab == 1).sum())}"
                          f", licit={int((lab == 0).sum())}"
                          f", unknown={int((lab == -1).sum())}")
            lines.append(f"  node[{nt}]: {len(df):,}{extra}")
        for rel, df in self.edges.items():
            lines.append(f"  edge{rel}: {len(df):,}")
        return "\n".join(lines)

    @staticmethod
    def _write(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
        """Parquet if available (much smaller/faster), else CSV."""
        try:
            df.to_parquet(out_dir / f"{stem}.parquet")
        except (ImportError, ValueError):
            df.to_csv(out_dir / f"{stem}.csv", index=False)

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for nt, df in self.nodes.items():
            self._write(df, out_dir, f"nodes_{nt}")
        for rel, df in self.edges.items():
            self._write(df, out_dir, f"edges_{'__'.join(rel)}")
        np.savez_compressed(
            out_dir / "arrays.npz",
            **{f"feat__{k}": v for k, v in self.features.items()},
            **{f"label__{k}": v for k, v in self.labels.items()},
            **{f"time__{k}": v for k, v in self.time.items()},
        )
        with open(out_dir / "summary.txt", "w") as f:
            f.write(self.summary() + "\n")


def temporal_split(time_array: np.ndarray, train_ratio: float = 0.7):
    """Split by time so we never train on the future. Returns boolean masks."""
    valid = time_array[time_array >= 0]
    if valid.size == 0:
        raise ValueError("no valid time steps")
    cutoff = np.quantile(valid, train_ratio)
    train = time_array <= cutoff
    test = time_array > cutoff
    return train, test


def random_split(n: int, train_ratio: float = 0.7, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * train_ratio)
    train = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    train[idx[:k]] = True
    test[idx[k:]] = True
    return train, test


def factorize_entity(series: pd.Series, prefix: str) -> tuple:
    """Map raw entity values -> contiguous integer node ids, dropping nulls."""
    s = series.astype("string")
    mask = s.notna() & (s.str.strip() != "")
    codes, uniques = pd.factorize(s[mask])
    nodes = pd.DataFrame({"node_id": [f"{prefix}:{u}" for u in uniques]})
    return codes, mask.to_numpy(), nodes
