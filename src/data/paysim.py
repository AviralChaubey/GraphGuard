"""PaySim mobile-money loader.

Builds an ACCOUNT graph (not a transaction graph): nameOrig -> nameDest edges.
This is the cleanest of the three for ring detection, because accounts are
already first-class and rings are literally connected components of accounts.

Expected: one CSV in data/raw/paysim/ with columns
  step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base import GraphBundle
from src.utils.config import get_logger, raw_dir

log = get_logger(__name__)


def load_paysim(root=None, nrows: int | None = None) -> GraphBundle:
    root = raw_dir("paysim") if root is None else root
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no CSV under {root}; see docs/DATASETS.md")
    log.info("reading paysim from %s", csvs[0].name)
    df = pd.read_csv(csvs[0], nrows=nrows)

    # account node table: union of origin and destination names
    accounts = pd.Index(pd.unique(pd.concat([df["nameOrig"], df["nameDest"]])))
    idx = pd.Series(np.arange(len(accounts)), index=accounts)
    df["src"] = df["nameOrig"].map(idx).astype(int)
    df["dst"] = df["nameDest"].map(idx).astype(int)

    # per-account aggregate features
    out = df.groupby("src").agg(out_cnt=("amount", "size"),
                                out_sum=("amount", "sum"),
                                out_mean=("amount", "mean"),
                                out_max=("amount", "max"))
    inc = df.groupby("dst").agg(in_cnt=("amount", "size"),
                                in_sum=("amount", "sum"),
                                in_mean=("amount", "mean"),
                                in_max=("amount", "max"))
    feat = (pd.DataFrame(index=np.arange(len(accounts)))
            .join(out).join(inc).fillna(0.0))
    # log-scale the heavy-tailed money columns
    for c in feat.columns:
        if c.endswith(("_sum", "_mean", "_max")):
            feat[c] = np.log1p(feat[c])
    feat["net_flow"] = feat["in_sum"] - feat["out_sum"]
    feat["degree"] = feat["in_cnt"] + feat["out_cnt"]

    # an account is fraudulent if it originated any fraudulent transaction
    fraud_src = df.loc[df["isFraud"] == 1, "src"].unique()
    y = np.zeros(len(accounts), dtype=int)
    y[fraud_src] = 1

    # account "time" = first step it appears in
    first_step = (pd.concat([df[["src", "step"]].rename(columns={"src": "n"}),
                             df[["dst", "step"]].rename(columns={"dst": "n"})])
                  .groupby("n")["step"].min())
    t = first_step.reindex(np.arange(len(accounts))).fillna(0).astype(int).to_numpy()

    b = GraphBundle(name="paysim", target_node_type="account")
    b.nodes["account"] = pd.DataFrame({"node_id": accounts.astype(str)})
    b.edges[("account", "sent_to", "account")] = df[["src", "dst"]].reset_index(drop=True)
    b.features["account"] = feat.to_numpy(dtype=np.float32)
    b.labels["account"] = y
    b.time["account"] = t
    return b
