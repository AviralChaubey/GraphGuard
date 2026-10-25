"""End-to-end check of the API against a REAL trained checkpoint (Workstream C).

    python scripts/verify_real_api.py --dataset elliptic
    python scripts/verify_real_api.py --dataset paysim

Exits 0 only if every check passes.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="elliptic", choices=["elliptic", "paysim"])
ap.add_argument("--checkpoint", default=None)
args = ap.parse_args()

ckpt = args.checkpoint or f"results/checkpoints/{args.dataset}/best.pt"
if not Path(ckpt).exists():
    sys.exit(f"FAIL: checkpoint not found: {ckpt} (train it first with scripts/train.py)")
os.environ["GRAPHGUARD_CHECKPOINT"] = ckpt
os.environ["GRAPHGUARD_DATASET"] = args.dataset

from fastapi.testclient import TestClient  # noqa: E402
from src.api.main import create_app  # noqa: E402

app = create_app()
client = TestClient(app)
state = app.state.gg
failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


r = client.get("/health")
check("health", r.status_code == 200 and r.json()["model_loaded"] is True, r.text)

# a few nodes, including ones whose true label is unknown (-1) -- these must NOT be rejected
y_true = state.data.y_true
unknown = [int(i) for i in (y_true == -1).nonzero().flatten()[:2]]
known = [int(i) for i in (y_true >= 0).nonzero().flatten()[:2]]
for idx in unknown + known:
    r = client.get(f"/score/{idx}")
    ok = r.status_code == 200 and 0 <= r.json()["fraud_score"] <= 1
    check(f"score/{idx}", ok, r.text[:120])

check("score out of range -> 404", client.get("/score/99999999").status_code == 404)

print(f"rings found: {len(state.rings)}")
if state.rings:
    rid = next(iter(state.rings))
    r = client.get(f"/ring/{rid}")
    check(f"ring/{rid}", r.status_code == 200 and r.json()["size"] >= 3, r.text[:120])
    r = client.get(f"/explain/{rid}")
    ok = r.status_code == 200 and len(r.json()["reason"]) > 0
    check(f"explain ring {rid}", ok, r.text[:200])
    if ok:
        print("  reason:", r.json()["reason"])
        print("  fidelity:", r.json()["fidelity"], "sparsity:", r.json()["sparsity"])
    r = client.post("/feedback", json={"ring_id": rid, "verdict": "confirmed", "note": "verify script"})
    check("feedback", r.status_code == 200, r.text[:120])
else:
    print("WARN no rings above threshold -- ring/explain-ring/feedback not exercised")

if known:
    r = client.get(f"/explain/{known[0]}")
    check(f"explain node {known[0]}", r.status_code == 200 and len(r.json()["reason"]) > 0, r.text[:200])

print("\nALL CHECKS PASSED" if not failures else f"\nFAILED: {failures}")
sys.exit(1 if failures else 0)
