"""FastAPI app for Workstream C.

Data/model wiring
------------------
Until A and B land their real loaders, this app runs against `DummyModel` and a single
synthetic star-shaped ring (`src/_stubs/`) so the dashboard and tests never have to wait --
see docs/team/PROMPT_C_explain_api_ui.md ("Build against DummyModel from day one").

To swap in the real thing once it exists, change only `_load_state()` below:
  - replace `DummyModel.load(...)` with `GraphGuardModel.load(checkpoint_path)` (from B)
  - replace `synthetic_star_graph()` with A's real `load_dataset()` + `to_pyg()` + B's
    `extract_rings()`
Nothing else in this file, in `src/explain/`, or in the dashboard needs to change, because
everything downstream is written against the frozen Contract 2 / Contract 3 shapes, not
against these stub implementations.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.explain.evidence import build_evidence_subgraph
from src.explain.gnnexplainer import GNNExplainerWrapper, get_node_labels
from src.explain.metrics import fidelity as compute_fidelity
from src.explain.metrics import sparsity as compute_sparsity
from src.explain.reason import ReasonContext, generate_reason
from src.api.schemas import (
    ExplainResponse,
    FeedbackAck,
    FeedbackRequest,
    HealthResponse,
    RingEdge,
    RingNode,
    RingResponse,
    ScoreResponse,
    TopFeature,
)
from src.utils.config import get_logger, load_config

logger = get_logger(__name__)
cfg = load_config()


@dataclass
class AppState:
    model: torch.nn.Module
    data: object                 # torch_geometric.data.Data
    rings: Dict[str, object]     # ring_id -> Ring
    explainer: GNNExplainerWrapper


def _load_state() -> AppState:
    """See module docstring: this is the one function to change once A/B land real code."""
    from src._stubs.dummy_model import DummyModel
    from src._stubs.ring import synthetic_star_graph

    model = DummyModel()
    data, ring = synthetic_star_graph()
    explainer = GNNExplainerWrapper(model)
    logger.warning(
        "API running against DummyModel + a synthetic ring -- replace _load_state() once "
        "A's data loader and B's trained model + extract_rings() are available."
    )
    return AppState(model=model, data=data, rings={ring.ring_id: ring}, explainer=explainer)


def create_app() -> FastAPI:
    app = FastAPI(title="GraphGuard API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.get("api", {}).get("cors_origins", ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.gg = _load_state()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=app.state.gg.model is not None)

    @app.get("/score/{node_idx}", response_model=ScoreResponse)
    def score(node_idx: int) -> ScoreResponse:
        state: AppState = app.state.gg
        _validate_scoreable_node(state, node_idx)
        with torch.no_grad():
            out = state.model(state.data)
        node_id = _node_id(state, node_idx)
        return ScoreResponse(
            node_id=node_id,
            fraud_score=float(out["fraud_score"][node_idx]),
            anomaly_score=float(out["anomaly_score"][node_idx]),
        )

    @app.get("/ring/{ring_id}", response_model=RingResponse)
    def get_ring(ring_id: str) -> RingResponse:
        state: AppState = app.state.gg
        ring = state.rings.get(ring_id)
        if ring is None:
            raise HTTPException(status_code=404, detail=f"unknown ring_id '{ring_id}'")

        with torch.no_grad():
            scores = state.model(state.data)["fraud_score"]

        nodes = [
            RingNode(
                id=n,
                node_id=_node_id(state, n),
                type=_node_type(state, n),
                score=float(scores[n]),
            )
            for n in ring.node_ids
        ]
        edges = [
            RingEdge(src=s, dst=d, relation="used_device", weight=1.0)
            for s, d in ring.edge_list
        ]
        return RingResponse(
            ring_id=ring.ring_id, score=ring.score, size=ring.size, pattern=ring.pattern,
            nodes=nodes, edges=edges,
        )

    @app.get("/explain/{target_id}", response_model=ExplainResponse)
    def explain(target_id: str) -> ExplainResponse:
        state: AppState = app.state.gg

        if target_id in state.rings:
            ring = state.rings[target_id]
            explainer_out = state.explainer.explain_ring(state.data, ring)
        else:
            try:
                node_idx = int(target_id)
            except ValueError:
                raise HTTPException(
                    status_code=404, detail=f"'{target_id}' is not a known ring_id or node index"
                )
            _validate_scoreable_node(state, node_idx)
            explainer_out = state.explainer.explain(state.data, node_idx)

        evidence = build_evidence_subgraph(
            edge_index=state.data.edge_index.numpy(),
            edge_mask=explainer_out.edge_mask,
            target_idx=explainer_out.target_idx,
            node_feature_mask=explainer_out.node_feature_mask,
        )

        full_edge_attr = getattr(state.data, "edge_attr", None)
        n_full_edges = state.data.edge_index.shape[1]

        def predict_fn(edge_index: np.ndarray) -> float:
            """Rerun the model with only `edge_index`'s edges present, for fidelity.

            `metrics.fidelity()` calls this exactly twice: once with the full graph's
            edge_index (unchanged), once with the evidence-only edge_index it builds from
            `evidence.edges`. We tell those two calls apart by edge count so that if the
            real model needs `edge_attr` (Contract 2 doesn't rule this out -- see the
            edge_attr passthrough in `_ModelAdapter`), the evidence-only call gets the
            correctly-sliced rows rather than dropping edge_attr and silently changing what
            the model sees.
            """
            ei = torch.as_tensor(edge_index, dtype=torch.long)
            edge_attr = None
            if full_edge_attr is not None:
                if ei.shape[1] == n_full_edges:
                    edge_attr = full_edge_attr
                else:
                    edge_attr = full_edge_attr[evidence.edge_ids]

            data_kwargs = {"x": state.data.x, "edge_index": ei}
            if edge_attr is not None:
                data_kwargs["edge_attr"] = edge_attr
            probe_data = type(state.data)(**data_kwargs)

            with torch.no_grad():
                return float(state.model(probe_data)["fraud_score"][explainer_out.target_idx])

        fid = compute_fidelity(
            predict_fn, state.data.edge_index.numpy(), evidence.edges
        ).fidelity
        spars = compute_sparsity(len(evidence.edges), state.data.edge_index.shape[1])

        reason_ctx = _build_reason_context(state, evidence)
        reason = generate_reason(evidence, reason_ctx)

        return ExplainResponse(
            target_id=target_id,
            reason=reason,
            evidence_nodes=evidence.nodes,
            evidence_edges=evidence.edges,
            top_features=[
                TopFeature(name=f"f{i}", importance=imp) for i, imp in evidence.top_features
            ],
            fidelity=fid,
            sparsity=spars,
        )

    @app.post("/feedback", response_model=FeedbackAck)
    def feedback(body: FeedbackRequest) -> FeedbackAck:
        """Record an analyst's verdict on a ring.

        NOTE -- scope of what this does today: this endpoint only appends the verdict to a
        JSONL log (`feedback.store_path` in configs/default.yaml). It is **not** yet wired
        into a retraining loop -- there's no scheduled job or script that reads this file
        back and feeds it into B's training pipeline. That integration is real, planned
        future work (see "Feedback loop" in docs/team/PROMPT_C_explain_api_ui.md's work
        queue), not something this endpoint already does. Don't describe this as "improving
        the model" to analysts or in the dashboard copy until that loop actually exists.
        """
        state: AppState = app.state.gg
        if body.ring_id not in state.rings:
            raise HTTPException(status_code=404, detail=f"unknown ring_id '{body.ring_id}'")

        store_path = cfg.get("feedback", {}).get("store_path", "results/feedback.jsonl")
        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
        record = {
            "ring_id": body.ring_id,
            "verdict": body.verdict,
            "note": body.note,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with open(store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("feedback stored: %s", record)
        return FeedbackAck(status="stored", ring_id=body.ring_id)

    return app


def _validate_scoreable_node(state: AppState, node_idx: int) -> None:
    """Range + entity-node validation, shared by `/score/{node_idx}` and the node branch of
    `/explain/{target_id}`.

    Per docs/team/PROMPT_C_explain_api_ui.md: entity nodes (card/device/address/email,
    label `-1`) are conduits, not things to classify -- they must never receive a fraud
    verdict or become a *direct* explanation target. (`/explain/{ring_id}` doesn't need this
    check: `GNNExplainerWrapper.explain_ring()` already excludes entity nodes when it picks
    which ring member to explain, so a ring's hub device can never slip through that path.)

    Raises:
        HTTPException(404) if `node_idx` is out of range for the current graph.
        HTTPException(422) if `node_idx` is in range but is an entity node -- 422 rather
            than 404 because the index itself is well-formed and does refer to a real node;
            it's just not a valid *target* for a fraud verdict or explanation.
    """
    if node_idx < 0 or node_idx >= state.data.x.size(0):
        raise HTTPException(status_code=404, detail=f"node index {node_idx} out of range")

    labels = get_node_labels(state.data)
    if labels is not None and int(labels[node_idx]) == -1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"node index {node_idx} is an entity node (label -1). Entity nodes "
                "(card/device/address/email) are conduits, not classification targets -- "
                "they cannot receive a fraud verdict or be a direct explanation target."
            ),
        )


def _node_id(state: AppState, idx: int) -> str:
    ids = getattr(state.data, "node_ids", None)
    return ids[idx] if ids else str(idx)


def _node_type(state: AppState, idx: int) -> str:
    types = getattr(state.data, "node_types", None)
    return types[idx] if types else "unknown"


def _build_reason_context(state: AppState, evidence) -> ReasonContext:
    """Best-effort context from whatever node metadata is on `state.data`.

    Once A's real `GraphBundle` is wired in, this is where entity-sharing counts and inflow
    stats over `inflow_window_hours` (see configs/default.yaml) get computed from the real
    transaction timestamps -- for now, with only a synthetic star graph, we can detect the
    shared hub entity but not a real inflow percentage.
    """
    labels = get_node_labels(state.data)
    node_id_map = {i: _node_id(state, i) for i in range(state.data.x.size(0))}
    node_type_map = {i: _node_type(state, i) for i in range(state.data.x.size(0))}

    shared_entity_idx: Optional[int] = None
    shared_entity_count: Optional[int] = None
    if labels is not None:
        entity_nodes = [n for n in evidence.nodes if int(labels[n]) == -1]
        if entity_nodes:
            shared_entity_idx = entity_nodes[0]
            shared_entity_count = sum(
                1 for s, d in evidence.edges if s == shared_entity_idx or d == shared_entity_idx
            )

    return ReasonContext(
        node_id_map=node_id_map,
        node_type_map=node_type_map,
        feature_names=None,  # None => report as anonymised/structural until A provides names
        shared_entity_idx=shared_entity_idx,
        shared_entity_count=shared_entity_count,
        inflow_pct=None,
        inflow_window_hours=cfg.get("explain", {}).get("inflow_window_hours"),
    )


app = create_app()
