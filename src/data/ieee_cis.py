"""IEEE-CIS loader — this is where we actually BUILD a graph from tabular data.

The dataset is flat rows. The whole GraphGuard thesis depends on turning those
rows into a heterogeneous graph:

    Transaction --USED_CARD-->    Card      (card1 + card2 + card3 + card4)
    Transaction --USED_DEVICE-->  Device    (DeviceInfo + DeviceType + id_30/31)
    Transaction --AT_ADDRESS-->   Address   (addr1 + addr2)
    Transaction --USED_EMAIL-->   Email     (P_emaildomain)

Two transactions that share a Card or a Device become 2 hops apart in the graph.
That 2-hop path IS the fraud-ring signal. One device fanning out to 40 cards is
a ring; a per-row model cannot see it, ours can.

Expected in data/raw/ieee_cis/: train_transaction.csv, train_identity.csv
(test_*.csv have no public labels — do not use them for evaluation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base import GraphBundle
from src.utils.config import get_logger, raw_dir

log = get_logger(__name__)

# raw column -> entity node type. Values are concatenated to form the entity key.
ENTITY_SPEC = {
    "card":    ["card1", "card2", "card3", "card5", "card4", "card6"],
    "device":  ["DeviceInfo", "DeviceType", "id_30", "id_31"],
    "address": ["addr1", "addr2"],
    "email":   ["P_emaildomain"],
}

# numeric transaction features we keep (the C/D/V blocks are huge; start small,
# M2 can widen this later)
BASE_NUMERIC = ["TransactionAmt", "TransactionDT", "dist1", "dist2"]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]


def _entity_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Join several raw columns into one entity identity string."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    key = df[present[0]].astype("string")
    for c in present[1:]:
        key = key.str.cat(df[c].astype("string"), sep="|", na_rep="?")
    # a row where every component was missing carries no information
    all_missing = df[present].isna().all(axis=1)
    key[all_missing] = pd.NA
    return key


def load_ieee_cis(root=None, nrows: int | None = None) -> GraphBundle:
    root = raw_dir("ieee_cis") if root is None else root
    tx_path = next(iter(root.rglob("train_transaction.csv")), None)
    id_path = next(iter(root.rglob("train_identity.csv")), None)
    if tx_path is None:
        raise FileNotFoundError(
            f"train_transaction.csv not found under {root}. "
            "Join the competition first — see docs/DATASETS.md."
        )

    log.info("reading %s", tx_path.name)
    tx = pd.read_csv(tx_path, nrows=nrows)
    if id_path is not None:
        log.info("joining %s (this is what gives us Device nodes)", id_path.name)
        ident = pd.read_csv(id_path)
        tx = tx.merge(ident, on="TransactionID", how="left")
    else:
        log.warning("train_identity.csv missing — no Device nodes will be built")

    n_tx = len(tx)
    log.info("%d transactions, %d columns", n_tx, tx.shape[1])

    b = GraphBundle(name="ieee_cis", target_node_type="transaction")

    # ---- transaction nodes -------------------------------------------------
    b.nodes["transaction"] = pd.DataFrame({"node_id": tx["TransactionID"].astype(str)})

    num_cols = [c for c in BASE_NUMERIC + C_COLS + D_COLS if c in tx.columns]
    X = tx[num_cols].copy()
    X["TransactionAmt"] = np.log1p(X["TransactionAmt"].clip(lower=0))
    # derived time features from TransactionDT (seconds from a reference point)
    if "TransactionDT" in X:
        X["hour_of_day"] = (tx["TransactionDT"] / 3600) % 24
        X["day_of_week"] = (tx["TransactionDT"] / 86400) % 7
        X = X.drop(columns=["TransactionDT"])
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    b.features["transaction"] = X.to_numpy(dtype=np.float32)

    b.labels["transaction"] = tx["isFraud"].astype(int).to_numpy()
    # day index as the temporal split key
    b.time["transaction"] = (tx["TransactionDT"] // 86400).astype(int).to_numpy()

    # ---- entity nodes + bipartite edges -----------------------------------
    for ent, cols in ENTITY_SPEC.items():
        key = _entity_key(tx, cols)
        mask = key.notna().to_numpy()
        if mask.sum() == 0:
            log.warning("entity '%s': no usable values, skipping", ent)
            continue
        codes, uniques = pd.factorize(key[mask])

        b.nodes[ent] = pd.DataFrame({"node_id": [f"{ent}:{u}" for u in uniques]})
        edges = pd.DataFrame({"src": np.flatnonzero(mask), "dst": codes})
        b.edges[("transaction", f"used_{ent}", ent)] = edges

        # entity features: how many transactions it touches, and their amounts.
        # fan-out is exactly the ring signal, so it must be a feature.
        amt = tx.loc[mask, "TransactionAmt"].to_numpy()
        g = pd.DataFrame({"e": codes, "amt": amt}).groupby("e")["amt"]
        feat = pd.DataFrame({
            "n_txn": g.size(),
            "amt_sum": np.log1p(g.sum()),
            "amt_mean": np.log1p(g.mean()),
            "amt_std": g.std().fillna(0.0),
        }).reindex(np.arange(len(uniques))).fillna(0.0)
        feat["fan_out"] = np.log1p(feat["n_txn"])
        b.features[ent] = feat.to_numpy(dtype=np.float32)
        b.labels[ent] = np.full(len(uniques), -1, dtype=int)   # entities are unlabelled
        b.time[ent] = np.zeros(len(uniques), dtype=int)

        log.info("entity '%s': %d nodes, %d edges (max fan-out %d)",
                 ent, len(uniques), len(edges), int(feat["n_txn"].max()))

    return b


def shared_entity_edges(bundle: GraphBundle, entity: str, max_group: int = 50) -> pd.DataFrame:
    """Optional: collapse transaction--entity--transaction into direct
    transaction<->transaction 'shares_entity' edges.

    Groups larger than `max_group` are skipped — a single huge hub (e.g. the
    'Windows' device string) would otherwise create millions of meaningless
    edges and blow up memory. Keep the bipartite form for those instead.
    """
    rel = ("transaction", f"used_{entity}", entity)
    if rel not in bundle.edges:
        raise KeyError(f"{rel} not in bundle")
    df = bundle.edges[rel]
    pairs = []
    for _, grp in df.groupby("dst")["src"]:
        v = grp.to_numpy()
        if 2 <= len(v) <= max_group:
            i, j = np.triu_indices(len(v), k=1)
            pairs.append(np.stack([v[i], v[j]], axis=1))
    if not pairs:
        return pd.DataFrame(columns=["src", "dst"])
    out = np.concatenate(pairs, axis=0)
    return pd.DataFrame(out, columns=["src", "dst"])
