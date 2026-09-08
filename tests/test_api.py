"""Integration tests for the FastAPI app. Requires torch + torch_geometric installed
(see requirements-c.txt) since these exercise the real GNNExplainer wrapper end-to-end --
no dataset or trained checkpoint needed, per docs/team/PROMPT_C_explain_api_ui.md.
"""
import pytest

torch_geometric = pytest.importorskip("torch_geometric")

from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_score_valid_and_invalid_node(client):
    r = client.get("/score/1")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["anomaly_score"] >= 0.0

    r_bad = client.get("/score/9999")
    assert r_bad.status_code == 404


def test_score_rejects_entity_node(client):
    # node 0 in the synthetic star graph is the hub device (label -1) -- an entity node,
    # never a valid fraud-score target.
    r = client.get("/score/0")
    assert r.status_code == 422
    assert "entity" in r.json()["detail"].lower()


def test_score_accepts_scoreable_node(client):
    # node 1 is a real 'card' account (label 1) -- must succeed where node 0 is rejected.
    r = client.get("/score/1")
    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == "card:1000|404"


def test_ring_shape_matches_contract(client):
    r = client.get("/ring/r_00042")
    assert r.status_code == 200
    body = r.json()
    for key in ("ring_id", "score", "size", "pattern", "nodes", "edges"):
        assert key in body
    assert body["nodes"][0].keys() >= {"id", "node_id", "type", "score"}
    assert body["edges"][0].keys() >= {"src", "dst", "relation", "weight"}


def test_ring_not_found(client):
    r = client.get("/ring/does_not_exist")
    assert r.status_code == 404


def test_explain_ring_shape_matches_contract(client):
    r = client.get("/explain/r_00042")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "target_id", "reason", "evidence_nodes", "evidence_edges",
        "top_features", "fidelity", "sparsity",
    ):
        assert key in body
    assert isinstance(body["reason"], str) and len(body["reason"]) > 0
    assert 0.0 <= body["fidelity"] <= 1.0
    assert 0.0 <= body["sparsity"] <= 1.0


def test_explain_single_node(client):
    r = client.get("/explain/1")
    assert r.status_code == 200


def test_explain_rejects_entity_node(client):
    # node 0 is the hub device (label -1) -- must never become a direct explanation target,
    # same rule as /score.
    r = client.get("/explain/0")
    assert r.status_code == 422
    assert "entity" in r.json()["detail"].lower()


def test_explain_unknown_target(client):
    r = client.get("/explain/not_a_ring_or_node")
    assert r.status_code == 404


def test_feedback_roundtrip(client, tmp_path, monkeypatch):
    import src.api.main as main_mod

    store_path = tmp_path / "feedback.jsonl"
    monkeypatch.setitem(main_mod.cfg.setdefault("feedback", {}), "store_path", str(store_path))

    r = client.post("/feedback", json={"ring_id": "r_00042", "verdict": "confirmed", "note": "ok"})
    assert r.status_code == 200
    assert r.json() == {"status": "stored", "ring_id": "r_00042"}
    assert store_path.exists()
    assert "confirmed" in store_path.read_text()


def test_feedback_unknown_ring_rejected(client):
    r = client.post("/feedback", json={"ring_id": "nope", "verdict": "dismissed"})
    assert r.status_code == 404
