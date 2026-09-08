# GraphGuard

**Explainable Graph Neural Networks for Fraud-Ring Detection**

Most fraud systems score one transaction at a time, so coordinated *rings* of accounts slip
through. GraphGuard models financial activity as a heterogeneous graph, detects the **ring**
rather than the individual account, and ships every alert with a **human-readable explanation**
an analyst or regulator can act on.

---

## Architecture

```
UI (React Dashboard)
        │  REST / WebSocket
        ▼
Backend Gateway (FastAPI)
        │
   ┌────┴─────────────┬──────────────────┐
   ▼                  ▼                  ▼
Graph Construction   GNN Inference     Explainability
& Feature Pipeline   Engine            Engine
   │                  │                  │
   └──────────────────┼──────────────────┘
                      ▼
        Graph Database Layer (Neo4j + PostgreSQL)
```

Full diagrams: `docs/ARCHITECTURE.md`

**Model core:** GraphSAGE (inductive encoder) → GAT (attention) → two heads
(supervised fraud classifier + autoencoder anomaly branch) → ring extraction → GNNExplainer.

---

## Repo structure

| Path | Contents |
|---|---|
| `src/data/` | Dataset loaders, cleaning, train/test splits |
| `src/graph/` | Graph builder, schema, feature engineering, Neo4j client |
| `src/models/` | GraphSAGE + GAT, autoencoder head, training loop |
| `src/ring/` | Ring extraction, community detection |
| `src/explain/` | GNNExplainer wrapper, evidence subgraph, reason templates |
| `src/api/` | FastAPI app and endpoints |
| `src/utils/` | Config loading, metrics, logging, seeding |
| `dashboard/` | React analyst dashboard |
| `notebooks/` | EDA and experiments (prefix with your initials) |
| `configs/` | YAML experiment configs |
| `scripts/` | One-shot scripts (download, train, evaluate) |
| `tests/` | Pytest unit tests |
| `results/` | Metrics, figures, checkpoints (gitignored) |
| `docs/` | Datasets, architecture, literature review |

---

## Setup

```bash
git clone <repo-url> && cd graphguard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_data.sh                        # see docs/DATASETS.md
```

Quickest possible start (no Kaggle account needed):

```python
from torch_geometric.datasets import EllipticBitcoinDataset
ds = EllipticBitcoinDataset(root="data/raw/elliptic")
```

---

## Datasets

Elliptic (Bitcoin) · IEEE-CIS (card fraud) · PaySim (mobile money).
Links, sizes and how to turn each into a graph: **`docs/DATASETS.md`**.
Raw data is gitignored — never commit it (IEEE-CIS competition terms forbid redistribution).

## Preprocessing

```bash
python scripts/preprocess.py --dataset elliptic
python scripts/preprocess.py --dataset ieee_cis --nrows 50000   # smoke run
```

Every loader returns a `GraphBundle` — the stable contract between workstreams A and B.
Details and the design decisions worth defending: **`docs/PREPROCESSING.md`**.

## Analysis

```bash
python scripts/analyze.py --dataset ieee_cis --nrows 200000
```

Writes fan-out histograms, degree distribution, fraud-rate-over-time and label-coverage
figures to `results/figures/`, plus a findings report to `results/metrics/`. The entity
fan-out figure is the empirical evidence that rings exist in the data — produced before
any model is trained.

## Evaluation

```bash
python scripts/evaluate.py --dataset elliptic --baselines   # runs today, no model needed
python scripts/evaluate.py --dataset elliptic --leakage     # temporal vs random split
```

PR-AUC is the headline metric, not accuracy. Metric choices, the two-tier baseline design
and how to read the leakage result: **`docs/EVALUATION.md`**.

## Tests

```bash
pytest tests/ -q          # or: python run_tests.py   (no pytest required)
```

Everything runs on synthetic fixtures — no dataset download, no trained checkpoint.

---

## Team & module ownership

| Workstream | Owner | Scope |
|---|---|---|
| **A** · Data & Graph | _TBD_ | `src/data/`, `src/graph/`, `notebooks/`, `scripts/preprocess.py` |
| **B** · Model & Ring | _TBD_ | `src/models/`, `src/ring/`, `scripts/train.py`, `scripts/evaluate.py` |
| **C** · Explainability, API & UI | _TBD_ | `src/explain/`, `src/api/`, `dashboard/` |

**Nobody edits another workstream's folder.** Ask the owner instead — that one rule prevents
almost every merge conflict. Interfaces between workstreams are frozen in
[`docs/CONTRACTS.md`](docs/CONTRACTS.md); per-person onboarding briefs are in `docs/team/`.

---

## Workflow

- `main` is protected — no direct pushes.
- Branch naming: `m2/gat-layers`, `m4/gnnexplainer-integration`.
- One PR per feature, one reviewer before merge.
- Notebooks are for exploration; anything reusable moves into `src/`.

---

## Evaluation

AUC · F1 · PR-AUC · Precision@k on all three datasets, plus ring purity, explanation
fidelity/sparsity, and end-to-end latency. Ablations: −GAT, −autoencoder, −ring layer.
