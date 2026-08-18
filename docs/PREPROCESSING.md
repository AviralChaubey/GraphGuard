# Preprocessing (M1)

## Run it

```bash
python scripts/preprocess.py --dataset elliptic
python scripts/preprocess.py --dataset ieee_cis --nrows 50000     # fast smoke run
python scripts/preprocess.py --dataset ieee_cis                   # full, ~5-10 min
python scripts/preprocess.py --dataset paysim --nrows 500000
```

Output lands in `data/processed/<dataset>/` (gitignored):
`nodes_*.parquet`, `edges_*.parquet`, `arrays.npz`, `splits.npz`, `summary.txt`.

## The contract between M1 and M2

Every loader returns a **`GraphBundle`** so nothing downstream cares which dataset it came from:

| Field | Meaning |
|---|---|
| `nodes[type]` | DataFrame with `node_id`, row index = integer node id |
| `edges[(src_type, relation, dst_type)]` | DataFrame with integer `src`, `dst` |
| `features[type]` | float32 array `[n_nodes, n_features]` |
| `labels[type]` | 1 = fraud, 0 = licit, **-1 = unknown** |
| `time[type]` | integer time-step, used for the temporal split |

M2 only ever calls `to_pyg(bundle)`. If M1 changes a loader internally, M2's code doesn't move.

## What each dataset becomes

**Elliptic** — already a graph. Transaction nodes, `sent_to` edges, 165 features, `time_step` 1–49.
~77% of labels are `-1`; that unlabelled mass is what the autoencoder branch (M3) exists for.

**PaySim** — we build an **account** graph, not a transaction graph, because rings are groups of
*accounts*. `nameOrig -> nameDest` becomes the edge; per-account aggregates (in/out count, sum,
mean, max, net flow, degree) become features. Money columns are `log1p`-scaled because the raw
amounts are heavy-tailed.

**IEEE-CIS** — the real work. Flat rows become a heterogeneous graph:

```
Transaction --used_card-->    Card     (card1|card2|card3|card5|card4|card6)
Transaction --used_device-->  Device   (DeviceInfo|DeviceType|id_30|id_31)
Transaction --used_address--> Address  (addr1|addr2)
Transaction --used_email-->   Email    (P_emaildomain)
```

Two transactions on the same device are now 2 hops apart. **That 2-hop path is the ring signal** —
one device fanning out to 40 cards is a ring, and a per-row model structurally cannot see it.
Entity nodes carry `n_txn`, `amt_sum/mean/std` and `fan_out` as features, since fan-out is
itself the strongest ring indicator.

`shared_entity_edges(bundle, "card")` optionally collapses this into direct transaction↔transaction
edges. It skips groups larger than `max_group` (default 50) — a hub like the `Windows` device
string would otherwise generate millions of meaningless edges.

## Decisions worth defending in the review

- **Temporal split, not random.** `temporal_split()` trains on early time-steps and tests on
  later ones. A random split leaks the future into training and inflates every metric. If your
  guide asks why your numbers look lower than a Kaggle notebook's, this is the answer.
- **Scaler fitted on train rows only.** `normalize_features(bundle, train_mask)` — fitting on
  everything leaks test statistics.
- **`-1` is not `0`.** Unknown-label nodes are excluded from the supervised loss, never treated
  as licit. Treating them as licit is a common and serious bug.
- **Missing values.** Numeric columns are median-filled; entity keys where *every* component is
  missing produce no node and no edge, rather than a fake "unknown" hub that would wrongly link
  thousands of unrelated transactions.
- **IEEE-CIS test files are unusable.** Labels were the competition's hidden answers. All
  evaluation happens on a split of `train_transaction.csv`.

## Order of work for M1

1. `python scripts/preprocess.py --dataset elliptic` — should run today, no Kaggle needed.
2. Sanity-check `summary.txt`: node/edge counts and the fraud rate.
3. IEEE-CIS with `--nrows 50000` first, then full.
4. Notebook EDA in `notebooks/` — degree distribution, fan-out histogram per entity type,
   fraud rate vs. time-step. The fan-out histogram is your best review slide: it shows rings
   exist in the data before any model is trained.
