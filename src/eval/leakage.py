"""The leakage experiment.

Runs an identical model under a temporal split and a random split, and reports
the PR-AUC inflation the random split buys.

Why this is worth doing: most published fraud results — and nearly every Kaggle
notebook — split randomly, which lets the model train on transactions that
occurred *after* the ones it is tested on. The gap measured here is the size of
that free lunch. It converts "our numbers look lower than the paper's" from an
apparent weakness into a quantified methodological finding.

Self-contained and cheap: one gradient-boosting model, two splits.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.base import GraphBundle, random_split, temporal_split
from src.eval.metrics import evaluate_detection
from src.utils.config import get_logger

log = get_logger(__name__)


def run_leakage_experiment(bundle: GraphBundle, node_type: str | None = None,
                           train_ratio: float = 0.7, seed: int = 42) -> dict:
    nt = node_type or bundle.target_node_type
    X, y, t = bundle.features[nt], bundle.labels[nt], bundle.time[nt]

    out = {}
    for name, (tr, te) in {
        "temporal": temporal_split(t, train_ratio),
        "random": random_split(len(y), train_ratio, seed),
    }.items():
        model = HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced",
            early_stopping=True, random_state=seed)
        m = tr & (y >= 0)
        model.fit(X[m], y[m])
        scores = model.predict_proba(X)[:, 1]
        out[name] = evaluate_detection(y, scores, mask=te)
        log.info("%-9s PR-AUC %.4f  ROC-AUC %.4f",
                 name, out[name].pr_auc, out[name].roc_auc)

    inflation = 100 * (out["random"].pr_auc / max(out["temporal"].pr_auc, 1e-9) - 1)
    out["inflation_pct"] = float(inflation)
    log.info("random split inflates PR-AUC by %.1f%%", inflation)
    return out


def format_report(res: dict) -> str:
    """Report what the numbers say — including when they don't show leakage.

    A negative result is a real result. If the random split scores LOWER, the
    honest reading is that the test period is intrinsically harder or the
    positive rate shifts over time, not that leakage was disproved. Never
    assert leakage the data doesn't support.
    """
    t, r = res["temporal"], res["random"]
    pct = res["inflation_pct"]

    if pct > 5:
        verdict = (
            f"  Random splitting inflates PR-AUC by {pct:+.1f}%.\n"
            "  That gap is look-ahead leakage: the model saw the future. We report\n"
            "  temporal numbers throughout, which is why ours sit below results\n"
            "  obtained under random splits.\n")
    elif pct < -5:
        verdict = (
            f"  The random split scores {abs(pct):.1f}% LOWER than the temporal one.\n"
            "  This does not disprove leakage — it means the two protocols are not\n"
            "  comparable here, most likely because the fraud rate shifts across\n"
            "  time-steps so the two test sets have different positive rates.\n"
            "  Report this as observed; do not claim a leakage effect from it.\n")
    else:
        verdict = (
            f"  Difference is small ({pct:+.1f}%) — no material leakage effect on\n"
            "  this dataset. We still use the temporal split, because it is the\n"
            "  protocol that matches deployment.\n")

    return (
        "Leakage experiment — identical model, two evaluation protocols\n"
        "-------------------------------------------------------------\n"
        f"  temporal split   PR-AUC {t.pr_auc:.4f}   ROC-AUC {t.roc_auc:.4f}   "
        f"F1 {t.f1_at_best:.4f}   pos-rate {t.positive_rate:.3%}\n"
        f"  random split     PR-AUC {r.pr_auc:.4f}   ROC-AUC {r.roc_auc:.4f}   "
        f"F1 {r.f1_at_best:.4f}   pos-rate {r.positive_rate:.3%}\n\n" + verdict)
