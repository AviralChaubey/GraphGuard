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
```

### Identity vs. category — why only two node types

We originally also modelled `addr1|addr2` and `P_emaildomain` as node types.
The EDA killed that. Measured on 200k transactions:

| entity | distinct | median fan-out | max fan-out | distinct/edges |
|---|---|---|---|---|
| card | 10,121 | 3 | 4,374 | 0.051 |
| device | 2,929 | 2 | 4,839 | 0.043 |
| address | 327 | 2 | 17,178 | **0.002** |
| email | 59 | 113 | 74,699 | **0.0004** |

An entity type earns a node type when many distinct values exist relative to
the transactions they touch — that is what makes sharing one *meaningful*.
`P_emaildomain` has 59 distinct values covering 84% of transactions: modelling
it as a node puts almost every transaction two hops from every other one, so
message passing carries no information. And "these accounts share gmail.com"
is not evidence of collusion. `addr1` is a region code, not a street address —
same problem.

Both are kept as **frequency-encoded transaction features** instead, so no
signal is lost, only the false adjacency. Frequency encoding, not target
encoding: target encoding leaks the label into the features and inflates every
metric.

`src/data/eda.py` computes this ratio automatically and warns when an entity
type looks category-like, so the same mistake can't be made silently with a
new column.

### What we tried, and why we abandoned it

`device` passes the identity test (2,929 distinct values over 68k edges), yet
naming its largest hubs showed:

```
device  4,839  Windows|desktop|Windows 10|chrome 63.0
device  4,314  Windows|desktop|?|chrome 63.0
device  3,038  Windows|desktop|Windows 7|chrome 63.0
```

That is a **browser configuration, not a device**. In 2017 "Windows 10 +
Chrome 63" was the most common desktop setup in the world. Keeping that edge
tells the model 4,839 strangers are connected — the opposite of the signal we
want, and it would have been our headline ring.

We first tried capping fan-out at 500, on the theory that an entity shared by
hundreds of transactions is a population rather than a ring. **It didn't
work.** The fan-out distribution is continuous, so the threshold only truncated
the tail: at a cap of 500 the largest surviving device was
`?|mobile|?|mobile safari 10.0` with 493 transactions — the same kind of thing,
one row lower. And the cap discarded 39% of card edges and 62% of device edges
for an arbitrary number.

The honest conclusion: **IEEE-CIS contains no device fingerprint.**
`DeviceInfo` is an OS/browser string and no threshold turns it into an
identity. The cap remains available as `max_entity_fanout` for ablations, but
is off by default. We keep all edges and let GAT attention learn that an edge
into a 4,000-transaction hub carries little information — which is what
attention is *for*. Fan-out is also an entity feature, so the model sees
hub-ness directly.

Cards, by contrast, checked out: `9500|321.0|150.0|226.0|visa|debit` — every
component present, a genuine key.

### Fraud concentration — the actual ring evidence

Fan-out measures **popularity**, and the most popular values are common browser
configurations. What identifies a ring is **concentration**: an entity whose
transactions are fraudulent far above the base rate.

**Ranking this correctly took three attempts, and the first two were wrong:**

| Rank by | What wins | Why it's wrong |
|---|---|---|
| fraud **count** | `Windows\|desktop\|?\|chrome 63.0` — 4,314 txns, 422 fraud (10%) | Rewards size. A big bucket with a slightly raised rate is not a ring. |
| fraud **rate** | any entity with 2 transactions, both fraud (100%) | Rewards noise. No evidence behind it. |
| **Wilson 95% lower bound** | 45 txns, 40 fraud (89%, LB 77%) | Demands a high rate *and* enough transactions to believe it. |

The Wilson lower bound asks: *given this much evidence, what is the worst the
true rate plausibly is?* A 4,314-transaction bucket at 10% has a lower bound of
9%; a 45-transaction entity at 89% has 77%. The tight ring wins, which is the
ordering the ring claim needs.

`fraud_concentration()` reports, per entity type, the entities with the highest
Wilson lower bound, along with rate and lift over base. The claim it produces —

> *one card key carries 41 transactions, 32 of them fraudulent — 78% against a
> 3.0% base rate, a 26x lift*

— survives someone asking what that entity actually is. *"One device has 4,839
transactions"* does not.

**Lesson worth keeping, and worth saying in the review:** a statistical test
said `device` was identity-like. Naming the top hubs revealed it wasn't. Then a
threshold looked like a fix, and checking the result showed it had only moved
the problem. The metric had to change, not the cutoff. Print your data at every
step; summary statistics hide exactly the thing you are looking for.

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
