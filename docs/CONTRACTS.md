# Interface Contracts

**This file is the law.** Anyone may propose a change to it, but nobody changes it alone —
a change here breaks someone else's code. Propose in the group, edit, commit the edit *first*,
then implement.

The point of these contracts: all three workstreams code against the signatures below from day
one, using stubs where the real thing doesn't exist yet. Nobody waits for anybody.

---

## Ownership map — who may edit what

| Path | Owner | Others may |
|---|---|---|
| `src/data/`, `src/graph/`, `notebooks/` | **A** | read only |
| `src/models/`, `src/ring/`, `scripts/train.py` | **B** | read only |
| `scripts/evaluate.py`, `scripts/analyze.py`, baselines & ablations | **A** | read only |
| `src/eval/` | **A** | read only |
| `src/explain/`, `src/api/`, `dashboard/` | **C** | read only |
| `src/utils/`, `configs/`, `docs/`, `README.md` | shared | edit — but announce it |
| `data/`, `results/` | nobody commits | gitignored |

If you need a change in someone else's folder, **ask them to make it**. Do not edit it yourself,
even if it's one line. That single rule prevents almost every merge conflict.

---

## Contract 1 — A → B  (`GraphBundle`)

Already implemented in `src/data/base.py`. **Frozen.**

```python
@dataclass
class GraphBundle:
    name: str
    nodes:    Dict[str, pd.DataFrame]     # node_type -> DataFrame with 'node_id'
    edges:    Dict[tuple, pd.DataFrame]   # (src_type, relation, dst_type) -> 'src','dst'
    features: Dict[str, np.ndarray]       # node_type -> float32 [n_nodes, n_feat]
    labels:   Dict[str, np.ndarray]       # 1 fraud, 0 licit, -1 unknown
    time:     Dict[str, np.ndarray]       # int time-step
    target_node_type: str
```

B consumes it only through:

```python
from src.graph import load_pyg

data = load_pyg("elliptic")     # loads the bundle AND its saved split
```

`load_pyg()` is the recommended entry point — it reads the split saved by
`scripts/preprocess.py`, so every experiment runs on byte-identical partitions.
Lower level, if you need to pass your own masks:

```python
from src.data import load_dataset
from src.graph import to_pyg
data = to_pyg(load_dataset("elliptic"), train_mask=..., test_mask=...)
```

### What `to_pyg()` returns — frozen

**Homogeneous** (Elliptic, PaySim) → `torch_geometric.data.Data` with:

| Attribute | Shape / meaning |
|---|---|
| `data.x` | `[n_nodes, n_features]` float32 |
| `data.y` | `[n_nodes]` long — **1 fraud, 0 licit, -1 unknown** |
| `data.edge_index` | `[2, n_edges]`, already undirected (reverse edges added) |
| `data.time` | `[n_nodes]` long — the time-step |
| `data.train_mask` | `[n_nodes]` bool — **temporal**: earlier time-steps |
| `data.test_mask` | `[n_nodes]` bool — later time-steps |
| `data.target_node_type` | str |

**Heterogeneous** (IEEE-CIS) → `torch_geometric.data.HeteroData`, **not** `Data`.
Same fields but per node type: `data['transaction'].x`, `.y`, `.train_mask`, `.test_mask`.
Entity types (`card`, `device`, `address`, `email`) carry `y = -1` and appear in
neither mask — they are conduits for message passing, never classified.
**B must handle both types**; `isinstance(data, HeteroData)` is the check.

**Guarantees A maintains:**

- Arrays are aligned to node index order.
- `-1` never appears in `train_mask` or `test_mask` — unknowns are excluded at
  mask-construction time, so a consumer cannot forget to.
- The masks are disjoint, and `time[train].max() <= time[test].min()` — the
  split is **temporal, never random**. This is enforced by a test.
- `to_pyg()` always adds reverse edges.
- A mask whose length doesn't match the bundle raises `ValueError` rather than
  silently mis-indexing.

---

## Contract 2 — B → C  (model + ring outputs)

B implements these exact signatures in `src/models/` and `src/ring/`. C writes against them
immediately using `DummyModel` (below) and swaps in the real one when it lands.

```python
# src/models/graphguard.py
class GraphGuardModel(torch.nn.Module):
    def forward(self, data) -> dict:
        """Returns:
        {
          'fraud_score':  Tensor [n_target_nodes]  in [0,1],
          'anomaly_score':Tensor [n_target_nodes]  reconstruction error, >= 0,
          'embeddings':   Tensor [n_target_nodes, hidden_dim],
          'attention':    Tensor [n_edges] or None  # per-edge GAT weight
        }"""

    @classmethod
    def load(cls, checkpoint_path: str) -> "GraphGuardModel": ...
```

```python
# src/ring/extract.py
@dataclass
class Ring:
    ring_id: str
    node_ids: list[int]          # indices into bundle.nodes[target_type]
    edge_list: list[tuple]       # [(src, dst), ...] internal edges
    score: float                 # 0-1 ring-level risk
    size: int
    pattern: str                 # 'star' | 'chain' | 'cyclic' | 'dense' | 'unknown'

def extract_rings(data, scores, threshold=0.7, min_size=3) -> list[Ring]: ...
```

```python
# src/models/dummy.py   <-- B writes this FIRST, day one, ~30 lines
class DummyModel:
    """Random but correctly-shaped outputs. Unblocks C immediately."""
```

**Rule:** `src/models/dummy.py` is committed on day one, before any real model work.
C is not allowed to be blocked waiting for B.

---

## Contract 3 — C → everyone  (API response shapes)

Frozen JSON. B and the dashboard both depend on these.

```jsonc
// GET /ring/{ring_id}
{
  "ring_id": "r_00042",
  "score": 0.91,
  "size": 7,
  "pattern": "star",
  "nodes": [{"id": 1234, "node_id": "card:13926|404", "type": "card", "score": 0.88}],
  "edges": [{"src": 1234, "dst": 5678, "relation": "used_device", "weight": 0.72}]
}

// GET /explain/{node_or_ring_id}
{
  "target_id": "r_00042",
  "reason": "Flagged because 6 accounts share device D-3391, and 82% of inflow ...",
  "evidence_nodes": [1234, 5678],
  "evidence_edges": [[1234, 5678]],
  "top_features": [{"name": "fan_out", "importance": 0.41}],
  "fidelity": 0.87,
  "sparsity": 0.12
}

// POST /feedback   {"ring_id": "r_00042", "verdict": "confirmed"|"dismissed", "note": "..."}
```

---

## Running the CLIs

`scripts/` is a package, so both forms work:

```bash
python scripts/evaluate.py --baselines
python -m scripts.evaluate --baselines
```

`--dataset` is optional and defaults to `data.dataset` in the config file, so
`python -m scripts.evaluate --checkpoint results/checkpoints/best.pt --config configs/default.yaml`
works as written. Pass `--dataset` explicitly to override it.

---

## Shared conventions — everyone follows these

- **Config, not constants.** Anything tunable goes in `configs/default.yaml`, read via
  `load_config()`. No magic numbers in code.
- **Logging, not print.** `from src.utils.config import get_logger`.
- **Seed everything.** Call `set_seed(cfg["seed"])` at the top of any script.
- **Type hints on every public function.** Docstring says what it returns.
- **One test per public function** in `tests/`, runnable with no dataset downloaded.
- **Never commit** `data/`, `results/checkpoints/`, `myenv/`.

## Git workflow

```
main                 protected, always working
  a/<feature>        e.g. a/paysim-balance-features
  b/<feature>        e.g. b/gat-layers
  c/<feature>        e.g. c/gnnexplainer-wrapper
```

```bash
git checkout main && git pull            # ALWAYS before starting
git checkout -b b/gat-layers
# ... work, commit often ...
git push -u origin b/gat-layers
# open a PR, one teammate reviews, then merge
```

Pull `main` at the start of every session. Merge to `main` at least twice a week — long-lived
branches are what cause painful conflicts, not the number of people.

## Definition of done (per feature)

1. Code in the right folder, respecting the contracts above
2. Test added, `pytest tests/` passes
3. Relevant `docs/*.md` updated
4. Config values in YAML, not hardcoded
5. PR reviewed by one teammate, merged to `main`
