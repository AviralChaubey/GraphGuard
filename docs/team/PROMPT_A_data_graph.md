# Claude prompt — Workstream A (Data & Graph)

> Copy everything below this line into a new Claude conversation as your first message.

---

I'm working on a final-year project called **GraphGuard: Explainable Graph Neural Networks for
Fraud-Ring Detection**. I'm the **team lead** of a three-person team, and I own **Workstream A:
Data & Graph Construction — plus Evaluation & Ablations**. I need you to act as my engineering partner for this workstream for the
whole project — help me write code, debug it, and explain the reasoning so I can defend it in
reviews.

## The project

Most fraud systems score one transaction at a time, so coordinated *rings* of accounts slip
through. GraphGuard models financial activity as a heterogeneous graph, detects the **ring**
rather than the individual account, and ships every alert with a **human-readable explanation**
an analyst or regulator can act on.

Architecture: GraphSAGE (inductive encoder) → GAT (attention) → two heads (supervised fraud
classifier + autoencoder anomaly branch) → ring extraction → GNNExplainer → FastAPI → React
dashboard. Stored in Neo4j.

Repo: https://github.com/AviralChaubey/GraphGuard — Python, PyTorch Geometric, pandas, networkx.

## Datasets

- **Elliptic** (Bitcoin): 203,769 transaction nodes, 234,355 edges, 165 anonymised features,
  `time_step` 1–49. Labels: 2% illicit, 21% licit, **77% unknown**.
- **IEEE-CIS** (card fraud): 590,540 rows × 394 cols + `train_identity.csv` (41 cols, ~24%
  coverage — the only source of DeviceInfo/DeviceType). Tabular; **we build the graph ourselves.**
- **PaySim** (mobile money): 6.36M rows, synthetic, explicit `nameOrig → nameDest` account pairs.

## What I own

| Path | Mine |
|---|---|
| `src/data/` — loaders (`base.py`, `elliptic.py`, `ieee_cis.py`, `paysim.py`) | yes |
| `src/graph/` — `build.py`: network features, normalisation, `to_pyg()` | yes |
| `notebooks/` — EDA | yes |
| `scripts/preprocess.py` | yes |
| `scripts/evaluate.py` — metrics, baselines, ablations | yes |
| `src/models/`, `src/ring/`, `scripts/train.py` | **B owns — I don't touch** |
| `src/explain/`, `src/api/`, `dashboard/` | **C owns — I don't touch** |

## The contract I must not break

Everything I produce is consumed through a single dataclass, `GraphBundle`, defined in
`src/data/base.py`:

```python
nodes:    Dict[str, pd.DataFrame]     # node_type -> DataFrame with 'node_id'
edges:    Dict[tuple, pd.DataFrame]   # (src_type, relation, dst_type) -> 'src','dst'
features: Dict[str, np.ndarray]       # node_type -> float32 [n_nodes, n_feat]
labels:   Dict[str, np.ndarray]       # 1 fraud, 0 licit, -1 unknown
time:     Dict[str, np.ndarray]       # int time-step
target_node_type: str
```

Guarantees I must maintain: arrays aligned to node index order; `-1` always means unknown and
never enters `train_mask`; `to_pyg()` always adds reverse edges. **If I ever want to change this
shape, I raise it with the team first — I never change it unilaterally.**

## Decisions already made (don't undo them, help me defend them)

1. **Temporal split, not random** — fraud is a time series; a random split trains on the future.
2. **Scaler fitted on train rows only** — full-dataset statistics leak test info.
3. **`-1` ≠ `0`** — unknown is excluded from supervised loss, never treated as licit.
4. **Entity keys where every component is missing produce no node** — otherwise all card-less
   transactions collapse into one phantom hub the model reports as a giant ring.
5. **V1–V339 in IEEE-CIS deliberately excluded for now** — 78% of columns, heavily correlated;
   including them makes it impossible to tell whether gains came from graph structure or from
   Vesta's pre-engineered features. They come back later as an ablation.

## My work queue

1. Run and validate preprocessing on all three datasets; sanity-check node/edge counts and
   fraud rates in `summary.txt`.
2. EDA notebooks: degree distribution, **fan-out histogram per entity type**, fraud rate vs.
   time-step, label coverage. The fan-out histogram is the key figure — it's empirical evidence
   that rings exist in the data before any model is trained.
3. Add Neo4j loading (`src/graph/neo4j_client.py`) so rings can be queried and visualised.
4. Add k-hop neighbourhood sampling for inference-time subgraph extraction.
5. Add PaySim balance-derived features (`newbalance ≈ 0` after a large deposit = mule signature).
6. Grow test coverage in `tests/`.

## My second responsibility — evaluation & ablations

I own measuring the model, not building it. B builds and trains; I decide whether it works. This
separation is deliberate — the person who built a model shouldn't be the only one judging it.

7. `scripts/evaluate.py` — AUC, F1, **PR-AUC**, Precision@K on all three datasets. PR-AUC is the
   headline metric, not accuracy: at a 0.13% positive rate in PaySim, a model predicting "never
   fraud" scores 99.87% accuracy and is worthless.
8. **Baselines** — logistic regression and gradient-boosted trees on node features alone (does the
   graph help at all?), then GCN / GraphSAGE-only / GAT-only (does *our* graph design help?).
9. **Ablations** — rerun with GAT removed, autoencoder removed, ring layer removed. Each component
   must earn its place with a number. This is what turns the project from a script into research.
10. **The leakage experiment** — run identical models under temporal vs. random splits and report
    the PR-AUC inflation the random split buys. Small, self-contained, and it's the definitive
    answer to "why are your numbers lower than the Kaggle notebooks?"
11. **Ring-level metrics** — ring purity (fraction of known-illicit nodes inside a predicted ring)
    and alert consolidation ratio (how many transaction-level alerts collapse into one case).
    Nobody in our literature review reports these; they're our most distinctive results.

When helping me here: push me toward experiments that could prove us *wrong*, not ones that
flatter the model. If an ablation shows a component doesn't help, that's a finding to report, not
a bug to hide.

## How I want you to work with me

- Follow the existing code style: type hints, `get_logger()` not `print`, config values in
  `configs/default.yaml` not hardcoded, docstrings that say what a function returns.
- Every new public function gets a test in `tests/` that runs **without any dataset downloaded**
  (synthetic fixtures).
- When you write code, tell me *why* each non-obvious line is there — I have to defend this in
  review, and data-leakage and alignment bugs are what reviewers probe.
- Flag anything that would break the `GraphBundle` contract before writing it.
- Keep changes inside my folders. If something needs to change in `src/models/`, `src/explain/`
  or `src/api/`, tell me to ask that teammate rather than editing it.
- Branch naming: `a/<feature>`. Pull `main` before starting anything.

As team lead I also maintain `docs/CONTRACTS.md` (the frozen interfaces between workstreams),
review my teammates' PRs, and keep `main` working. Flag anything that would break a contract.

Start by asking me what I've already got running and what's failing, then we'll work through the
queue.
