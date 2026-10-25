"""Pydantic models for the frozen API contract (docs/CONTRACTS.md, Contract 3).

These shapes are frozen -- B and the dashboard both depend on them. If they need to change,
raise it with the team first; don't change them unilaterally (see
docs/team/PROMPT_C_explain_api_ui.md).
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class RingNode(BaseModel):
    id: int
    node_id: str
    type: str
    score: float


class RingEdge(BaseModel):
    src: int
    dst: int
    relation: str
    weight: float


class RingResponse(BaseModel):
    ring_id: str
    score: float
    size: int
    pattern: str
    nodes: List[RingNode]
    edges: List[RingEdge]


class TopFeature(BaseModel):
    name: str
    importance: float


class ExplainResponse(BaseModel):
    target_id: str
    reason: str
    evidence_nodes: List[int]
    evidence_edges: List[Tuple[int, int]]
    top_features: List[TopFeature]
    fidelity: float
    sparsity: float


class ScoreResponse(BaseModel):
    node_id: str
    fraud_score: float
    anomaly_score: float


class FeedbackRequest(BaseModel):
    ring_id: str
    verdict: Literal["confirmed", "dismissed"]
    note: Optional[str] = None


class FeedbackAck(BaseModel):
    status: Literal["stored"]
    ring_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class RingSummary(BaseModel):
    """One row of GET /rings -- enough to populate an alert queue without fetching every
    ring's full node/edge list. Added because the dashboard previously had no way to discover
    which ring_ids actually exist on a given running server (see the note in
    AlertQueue.jsx) and was relying on a hardcoded list computed elsewhere -- which breaks the
    moment ring extraction produces even slightly different scores (different machine, OS,
    PyTorch build -- floating point is not guaranteed bit-identical across those), because the
    hardcoded ids may not exist in THIS server's actual rings."""
    ring_id: str
    score: float
    size: int
    pattern: str


class RingsResponse(BaseModel):
    rings: List[RingSummary]
    total: int
