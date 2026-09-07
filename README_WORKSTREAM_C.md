# Workstream C — Explainability, API & Dashboard

This is only my part of the GraphGuard repo (`src/explain/`, `src/api/`, `dashboard/`). It runs
standalone against `DummyModel` and a synthetic ring, per `docs/team/PROMPT_C_explain_api_ui.md`
— you don't need A's real dataset or B's trained model to use any of this.

```
src/
  utils/config.py      shared load_config()/get_logger()/set_seed() -- drop once a teammate's
                        real src/utils/ lands, see the note at the top of the file
  explain/
    gnnexplainer.py     wraps PyG's GNNExplainer against the frozen model.forward() contract
    evidence.py          mask -> minimal, readable evidence subgraph
    reason.py             evidence -> deterministic, rule-based natural-language reason
    metrics.py             fidelity + sparsity
  api/
    main.py               FastAPI app: /health /score /ring/{id} /explain/{id} /feedback
    schemas.py             pydantic models for the frozen Contract 3 JSON
  _stubs/                  DummyModel + a synthetic ring, standing in for B's real code --
                            delete once src/models/dummy.py and src/ring/extract.py exist
dashboard/                React analyst console (alert queue, ring graph, explanation panel)
tests/                     pytest, no dataset/checkpoint required
```

## 1. Backend — setup

```bash
cd graphguard
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-c.txt
```

## 2. Backend — run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then check it's alive:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ring/r_00042
curl http://localhost:8000/explain/r_00042
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"ring_id":"r_00042","verdict":"confirmed","note":"looks right"}'
```

`r_00042` is the id of the one synthetic star-shaped ring baked into `src/_stubs/ring.py` —
that's the demo data until A/B's real pipeline is wired into `_load_state()` in
`src/api/main.py` (see the docstring at the top of that file for exactly what to change).

## 3. Backend — run the tests

```bash
pytest tests/ -v
```

`tests/test_evidence.py`, `test_reason.py`, and `test_metrics.py` are pure numpy/python and
run instantly with no extra setup. `tests/test_api.py` exercises the real `GNNExplainer`
end-to-end through `TestClient`, so it needs `torch` + `torch_geometric` installed
(`pip install -r requirements-c.txt`) — it's skipped automatically if they're missing.

## 4. Dashboard — setup & run

Needs Node 18+.

```bash
cd dashboard
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`). It expects the API from step 2
running at `http://localhost:8000` — override with a `.env` file containing
`VITE_API_BASE_URL=http://your-host:port` if needed.

## Swapping in the real model/data

Everything here is written against the frozen contracts in `docs/CONTRACTS.md`, not against
the stubs — so when B's real `GraphGuardModel` + `extract_rings()` and A's real
`load_dataset()` land, the only code that needs to change is `_load_state()` at the top of
`src/api/main.py`. Nothing in `src/explain/` or `dashboard/` should need to change.

## Things worth knowing when demoing this

- **IEEE-CIS vs Elliptic:** explanations are only as readable as the node IDs and feature names
  underneath them. IEEE-CIS has human-meaningful IDs (`card:13926|404`) so `reason.py` can name
  the shared entity; Elliptic's features are anonymised, so on that dataset you'll only ever see
  the structural fallback reason. That's a property of the data, not a bug in this code.
- **Why GNNExplainer over raw attention, and over SHAP** — see the docstring at the top of
  `src/explain/gnnexplainer.py`.
- **Why reasons are templated, never LLM-generated** — see the docstring at the top of
  `src/explain/reason.py`. This is the whole point of the workstream: an LLM-written
  explanation would itself be unauditable.
- **Fidelity / sparsity** are computed for every `/explain/` call, not just shown in the demo —
  they're meant to go straight into the paper's results section.
- **Feedback is persisted only, not yet a retraining loop.** `POST /feedback` appends the
  analyst's verdict to `feedback.store_path` (a JSONL file) and nothing else. There's no
  scheduled job that reads it back into B's training pipeline yet — that's still open work
  from the "Feedback loop" item in `docs/team/PROMPT_C_explain_api_ui.md`. Don't describe it
  to analysts as already improving the model.
- **DummyModel is a real (untrained) GraphSAGE, not `torch.rand(...)`.** Its `fraud_score`
  genuinely depends on node features and graph connectivity through PyG `MessagePassing`
  layers, and its weights are fixed at construction — so it's deterministic and gives
  GNNExplainer real gradients to optimise against. See the docstring in
  `src/_stubs/dummy_model.py` for why the earlier random version was actively wrong, not
  just low-fidelity.
- **`explain_ring()` never selects an entity node as the explanation target.** Entity nodes
  (card/device/address/email, label `-1`) are excluded when picking the ring's
  highest-scoring member — see the docstring in `GNNExplainerWrapper.explain_ring()`.
