# Claude prompt — Workstream B (Model & Ring Detection)

> Copy everything below this line into a new Claude conversation as your first message.

---

I'm working on a final-year project called **GraphGuard: Explainable Graph Neural Networks for
Fraud-Ring Detection**. I'm one of three people on the team, and I own **Workstream B: the GNN
model and ring extraction** — the core of the project. I need you to act as my engineering
partner for this workstream for the whole project — help me write code, debug training, and
explain the reasoning so I can defend it in reviews.

## The project

Most fraud systems score one transaction at a time, so coordinated *rings* of accounts slip
through. GraphGuard models financial activity as a heterogeneous graph, detects the **ring**
rather than the individual account, and ships every alert with a **human-readable explanation**.

My part of the architecture:

```
GraphSAGE encoder (inductive)  →  GAT attention layers  →  ┬→ Head A: supervised fraud classifier
                                                            └→ Head B: autoencoder (novel anomalies)
                                                                    ↓
                                                         Ring extraction (community detection)
```

Repo: https://github.com/AviralChaubey/GraphGuard — PyTorch + PyTorch Geometric.

## Why this architecture (from the papers we cite)

- **GraphSAGE / inductive** — Van Belle et al. 2022 (ESWA): inductive embeddings generalise to
  new accounts with no retraining, generated in <3 ms, ~40% PR-AUC lift. Fraud is live; a
  transductive model can't score an account created five seconds ago.
- **GAT / attention** — Sheu & Li 2022 (J. Money Laundering Control): validated GAT on exactly
  our problem and datasets. Attention weights each neighbour rather than averaging, which matters
  because fraudsters camouflage by connecting to many legitimate accounts (heterophily —
  see Ren et al. 2023, KBS).
- **Group-level output** — Cheng et al. 2023 (IEEE TKDE, GAGNN): models fraud as *gangs*, with
  Star / Chain / Cyclic laundering topologies. We reuse that typology.
- **The gap we fill** — none of these combines ring-level detection + explainability + validation
  on public reproducible benchmarks. Li, Jia & Su 2025 (Scientific Reports) is the accuracy bar
  on Elliptic: 0.9814 precision, 0.905 F1 — and it has no explanation and no ring output.

## What I own

| Path | Mine |
|---|---|
| `src/models/` — encoder, GAT, dual heads, training loop | yes |
| `src/ring/` — ring extraction, community detection | yes |
| `scripts/train.py` | yes |
| `src/data/`, `src/graph/`, `scripts/evaluate.py` | **A owns — I don't touch** |
| `src/explain/`, `src/api/`, `dashboard/` | **C owns — I don't touch** |

## The contracts I must honour

**Input, from A** — I only ever consume data through:

```python
from src.data import load_dataset
from src.graph import to_pyg
bundle = load_dataset("elliptic")
data   = to_pyg(bundle)      # torch_geometric Data or HeteroData
```

Labels are `1` fraud, `0` licit, **`-1` unknown** — `-1` must be excluded from the supervised
loss, never treated as licit (77% of Elliptic is `-1`). Splits come from `splits.npz` and are
**temporal** — train on the past, test on the future. My numbers will look lower than random-split
Kaggle notebooks; that is correct and I should say so rather than "fix" it.

**Output, to C** — I must implement exactly these signatures:

```python
# src/models/graphguard.py
class GraphGuardModel(torch.nn.Module):
    def forward(self, data) -> dict:
        """{'fraud_score': Tensor[n] in [0,1],
            'anomaly_score': Tensor[n] >= 0,
            'embeddings': Tensor[n, hidden_dim],
            'attention': Tensor[n_edges] or None}"""
    @classmethod
    def load(cls, checkpoint_path: str) -> "GraphGuardModel": ...

# src/ring/extract.py
@dataclass
class Ring:
    ring_id: str; node_ids: list[int]; edge_list: list[tuple]
    score: float; size: int
    pattern: str   # 'star' | 'chain' | 'cyclic' | 'dense' | 'unknown'

def extract_rings(data, scores, threshold=0.7, min_size=3) -> list[Ring]: ...
```

**Day-one obligation:** before any real model work I commit `src/models/dummy.py` — a ~30-line
`DummyModel` returning random but correctly-shaped outputs, so C can build the explainability and
API layers immediately instead of waiting for me. This is non-negotiable; C must never be blocked.

## My work queue

1. `src/models/dummy.py` — first commit, unblocks C.
2. GraphSAGE encoder + GAT layers, homogeneous first (Elliptic), then `HeteroConv` for IEEE-CIS.
3. Training loop with **weighted loss** for class imbalance (Elliptic ~2%, IEEE-CIS 3.5%,
   PaySim 0.13%), early stopping, checkpointing to `results/checkpoints/`.
4. Autoencoder head trained on normal-behaviour nodes only; reconstruction error = anomaly score.
5. Ring extraction: threshold high-scoring nodes → induced subgraph → Louvain communities →
   classify each into Star / Chain / Cyclic / Dense by its topology.
6. Expose config flags that let A disable components independently (`--no-gat`,
   `--no-autoencoder`, `--no-ring`) — **A owns evaluation and ablations**, and needs to be able to
   run my model with pieces switched off without editing my code. Build for that from the start.
7. Checkpoints saved with the config that produced them, so any result A reports is reproducible.

**A measures my model; I don't grade my own work.** If A's ablations show a component doesn't
help, that's a finding we report, not a number to tune until it looks better.

## How I want you to work with me

- Follow existing style: type hints, `get_logger()` not `print`, all hyperparameters in
  `configs/default.yaml` via `load_config()`, `set_seed()` at the top of every script.
- Tests in `tests/` that run on tiny synthetic graphs with **no dataset downloaded** — assert
  output shapes and value ranges, not accuracy.
- Explain *why* for anything non-obvious. I'll be asked "why GAT and not GCN", "why is your F1
  lower than the paper's", "why weighted loss instead of SMOTE" — help me answer those properly.
- Never silently treat `-1` as `0`. Never switch to a random split to make numbers look better.
- Keep changes inside my folders. If something needs changing in `src/data/` or `src/graph/`,
  tell me to ask A rather than editing it.
- Branch naming: `b/<feature>`. Pull `main` before starting anything.

Start by asking me what stage I'm at and whether preprocessing output exists yet, then we'll
start with `dummy.py` and the encoder.
