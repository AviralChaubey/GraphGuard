"""Elliptic Bitcoin dataset loader.

Expected files in data/raw/elliptic/:
  elliptic_txs_features.csv   (no header: txId, time_step, 165 features)
  elliptic_txs_classes.csv    (header: txId, class)  class: '1'=illicit, '2'=licit, 'unknown'
  elliptic_txs_edgelist.csv   (header: txId1, txId2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base import GraphBundle
from src.utils.config import get_logger, raw_dir

log = get_logger(__name__)

N_FEATURES = 165  # 93 local + 72 aggregated


def load_elliptic(root=None) -> GraphBundle:
    root = raw_dir("elliptic") if root is None else root
    # some Kaggle unzips nest the files one level deep
    if not (root / "elliptic_txs_features.csv").exists():
        nested = list(root.rglob("elliptic_txs_features.csv"))
        if not nested:
            raise FileNotFoundError(
                f"elliptic_txs_features.csv not found under {root}. "
                "See docs/DATASETS.md for the download step."
            )
        root = nested[0].parent

    log.info("reading elliptic from %s", root)
    cols = ["txId", "time_step"] + [f"f{i}" for i in range(N_FEATURES)]
    feats = pd.read_csv(root / "elliptic_txs_features.csv", header=None, names=cols)
    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    edges = pd.read_csv(root / "elliptic_txs_edgelist.csv")

    # txId -> contiguous index
    feats = feats.reset_index(drop=True)
    idx = pd.Series(feats.index.values, index=feats["txId"].values)

    # labels: 1 illicit -> 1, 2 licit -> 0, unknown -> -1
    lab_map = {"1": 1, "2": 0, 1: 1, 2: 0}
    classes["y"] = classes["class"].map(lab_map).fillna(-1).astype(int)
    y = (classes.set_index("txId")["y"].reindex(feats["txId"]).fillna(-1)
         .astype(int).to_numpy())

    # edges -> integer indices, drop any node not present in features
    e = edges.copy()
    e["src"] = e["txId1"].map(idx)
    e["dst"] = e["txId2"].map(idx)
    before = len(e)
    e = e.dropna(subset=["src", "dst"]).astype({"src": int, "dst": int})
    if len(e) < before:
        log.warning("dropped %d edges with unknown endpoints", before - len(e))

    X = feats[[f"f{i}" for i in range(N_FEATURES)]].to_numpy(dtype=np.float32)
    t = feats["time_step"].to_numpy(dtype=int)

    b = GraphBundle(name="elliptic", target_node_type="transaction")
    b.nodes["transaction"] = pd.DataFrame({"node_id": feats["txId"].astype(str)})
    b.edges[("transaction", "sent_to", "transaction")] = e[["src", "dst"]].reset_index(drop=True)
    b.features["transaction"] = X
    b.labels["transaction"] = y
    b.time["transaction"] = t
    return b
