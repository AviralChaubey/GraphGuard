# Claude prompt — Workstream C (Explainability, API & Dashboard)

> Copy everything below this line into a new Claude conversation as your first message.

---

I'm working on a final-year project called **GraphGuard: Explainable Graph Neural Networks for
Fraud-Ring Detection**. I'm one of three people on the team, and I own **Workstream C:
explainability, the API, and the analyst dashboard** — the part that makes this project *novel*
rather than another GNN fraud classifier. I need you to act as my engineering partner for this
workstream for the whole project.

## The project

Most fraud systems score one transaction at a time, so coordinated *rings* of accounts slip
through. GraphGuard detects the **ring** rather than the individual account — and crucially,
ships every alert with a **human-readable explanation** an analyst or regulator can act on.

**Why my part is the contribution:** every recent paper we reviewed (Sheu & Li 2022,
Van Belle 2022, Ren 2023, Cheng 2023, Li/Jia/Su 2025) produces a fraud score and stops there.
None explains itself. That matters practically — rule-based AML systems have false-positive rates
above 90%, so analysts drown — and legally: under GDPR-style "right to explanation" and RBI/FATF
audit expectations, a bank must justify why an account was frozen. **A black-box score cannot be
deployed in a regulated institution.** My layer is what makes the model usable at all.

## My part of the architecture

```
model outputs ─→ GNNExplainer ─→ Evidence Subgraph ─→ Reason Generator ─→ FastAPI ─→ React dashboard
                                                                                          │
                                                                    analyst feedback ─────┘
```

Repo: https://github.com/AviralChaubey/GraphGuard — Python, PyTorch Geometric, FastAPI, React.

## What I own

| Path | Mine |
|---|---|
| `src/explain/` — GNNExplainer wrapper, evidence subgraph, reason templates | yes |
| `src/api/` — FastAPI app and endpoints | yes |
| `dashboard/` — React analyst UI | yes |
| `src/data/`, `src/graph/` | **A owns — I don't touch** |
| `src/models/`, `src/ring/` | **B owns — I don't touch** |

## The contracts I code against

**From B** — these signatures are frozen, and B commits `src/models/dummy.py` on day one so I am
**never blocked waiting for a trained model**. I build everything against `DummyModel` first.

```python
model.forward(data) -> {
  'fraud_score':   Tensor[n] in [0,1],
  'anomaly_score': Tensor[n] >= 0,
  'embeddings':    Tensor[n, hidden_dim],
  'attention':     Tensor[n_edges] or None,
}

@dataclass
class Ring:
    ring_id: str; node_ids: list[int]; edge_list: list[tuple]
    score: float; size: int
    pattern: str   # 'star' | 'chain' | 'cyclic' | 'dense' | 'unknown'
```

**From me — frozen JSON the dashboard and B both depend on:**

```jsonc
// GET /ring/{ring_id}
{"ring_id":"r_00042","score":0.91,"size":7,"pattern":"star",
 "nodes":[{"id":1234,"node_id":"card:13926|404","type":"card","score":0.88}],
 "edges":[{"src":1234,"dst":5678,"relation":"used_device","weight":0.72}]}

// GET /explain/{node_or_ring_id}
{"target_id":"r_00042",
 "reason":"Flagged because 6 accounts share device D-3391, and 82% of inflow ...",
 "evidence_nodes":[1234,5678],"evidence_edges":[[1234,5678]],
 "top_features":[{"name":"fan_out","importance":0.41}],
 "fidelity":0.87,"sparsity":0.12}

// POST /feedback  {"ring_id":"r_00042","verdict":"confirmed"|"dismissed","note":"..."}
```

If I need to change these shapes, I raise it with the team first — I never change them alone.

## Things I need to know about the data

- Node IDs are human-meaningful in **IEEE-CIS** (`card:13926|404`, `device:SM-G930V|mobile`) —
  that's what makes readable explanations possible.
- In **Elliptic** features are anonymised (`f47` means nothing to anyone), so explanations there
  can only be structural: "these edges mattered". This is a property of the data, not a flaw in
  my method — **IEEE-CIS is where I demo the explainability story.**
- Entity nodes (card, device, address, email) have label `-1` — they're conduits, not things to
  classify. Explanations point *at* them; they never get a fraud verdict of their own.

## My work queue

1. `src/explain/gnnexplainer.py` — wrap PyG's `GNNExplainer`/`Explainer`, take a node or ring,
   return the edge mask and feature mask.
2. `src/explain/evidence.py` — turn masks into a minimal evidence subgraph (top-k edges + the
   nodes they touch). Small enough for a human to read.
3. `src/explain/reason.py` — template natural language from the evidence:
   *"Flagged because {n} accounts share {entity}, and {pct}% of inflow in {hours}h came from that
   cluster."* Rule-based templates, not an LLM — must be deterministic and auditable.
4. `src/explain/metrics.py` — **fidelity** (does removing the evidence subgraph change the
   prediction?) and **sparsity** (is it small enough to read?). These are quantitative results
   for the paper, not just a demo.
5. `src/api/main.py` — FastAPI: `/score`, `/ring/{id}`, `/explain/{id}`, `/feedback`, `/health`.
6. `dashboard/` — React: alert queue, ring graph view (Cytoscape.js or react-force-graph),
   explanation panel, confirm/dismiss buttons.
7. Feedback loop: store analyst verdicts so they can re-enter training.

## How I want you to work with me

- Follow existing style: type hints, `get_logger()` not `print`, config in
  `configs/default.yaml` via `load_config()`, docstrings saying what a function returns.
- Tests in `tests/` using `DummyModel` and synthetic graphs — **no dataset or trained checkpoint
  required**. The API must be testable with FastAPI's `TestClient`.
- Build against `DummyModel` from day one. Never write code that only runs once B finishes.
- Explanations must be **deterministic and rule-based**. If we generated reasons with an LLM, the
  explanation itself becomes a black box and the entire contribution collapses. Say so if I'm
  ever tempted.
- Explain *why* for anything non-obvious — I'll be asked "how is GNNExplainer different from just
  showing attention weights", "how do you know the explanation is faithful", "why not SHAP".
- Keep changes inside my folders. If something needs to change in `src/models/` or `src/data/`,
  tell me to ask that teammate rather than editing it.
- Branch naming: `c/<feature>`. Pull `main` before starting anything.

Start by asking me what exists in the repo so far, then we'll begin with the GNNExplainer wrapper
against `DummyModel`.
