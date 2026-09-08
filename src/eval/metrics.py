"""Detection and ring-quality metrics.

Design rules enforced here:
  * PR-AUC is the headline, not accuracy. At a 0.13% positive rate a model that
    predicts "never fraud" scores 99.87% accuracy and is worthless.
  * Unknown labels (-1) are excluded from every metric. Scoring them as licit
    would silently reward a model for calling unexamined nodes clean.
  * Precision@K is reported because it is what an analyst actually experiences:
    they work a fixed-length queue each day, not a probability distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)


@dataclass
class DetectionMetrics:
    n_evaluated: int
    n_positive: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    f1_at_best: float
    best_threshold: float
    f1_at_half: float = 0.0
    precision_at_k: dict = field(default_factory=dict)
    recall_at_k: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        pk = "  ".join(f"P@{k}={v:.3f}" for k, v in self.precision_at_k.items())
        return (f"n={self.n_evaluated:,} pos={self.n_positive:,} "
                f"({self.positive_rate:.3%})\n"
                f"PR-AUC={self.pr_auc:.4f}  ROC-AUC={self.roc_auc:.4f}  "
                f"F1={self.f1_at_best:.4f} @thr={self.best_threshold:.3f}  "
                f"F1@0.5={self.f1_at_half:.4f}\n{pk}")


def evaluate_detection(y_true, y_score, mask=None,
                       ks=(50, 100, 500, 1000)) -> DetectionMetrics:
    """Core detection metrics on labelled nodes only."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    keep = y_true >= 0                       # drop unknowns, always
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    y, s = y_true[keep], y_score[keep]

    if len(y) == 0 or y.sum() == 0:
        raise ValueError("no labelled positives in the evaluation set — "
                         "check the split; a temporal split can strand all "
                         "positives on one side")

    prec, rec, thr = precision_recall_curve(y, s)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best = int(np.nanargmax(f1s))
    best_thr = float(thr[min(best, len(thr) - 1)]) if len(thr) else 0.5

    order = np.argsort(-s)
    p_at_k, r_at_k = {}, {}
    for k in ks:
        if k > len(y):
            continue
        top = order[:k]
        hits = int(y[top].sum())
        p_at_k[k] = hits / k
        r_at_k[k] = hits / int(y.sum())

    # F1 at the deployed operating point, not just the best achievable one
    f1_half = float(f1_score(y, (s >= 0.5).astype(int), zero_division=0))

    return DetectionMetrics(
        n_evaluated=int(len(y)), n_positive=int(y.sum()),
        positive_rate=float(y.mean()),
        roc_auc=float(roc_auc_score(y, s)),
        pr_auc=float(average_precision_score(y, s)),
        f1_at_best=float(f1s[best]),
        best_threshold=best_thr,
        f1_at_half=f1_half,
        precision_at_k=p_at_k, recall_at_k=r_at_k)


# --------------------------------------------------------------------------
@dataclass
class RingMetrics:
    n_rings: int
    mean_size: float
    mean_purity: float
    weighted_purity: float
    consolidation_ratio: float
    coverage: float

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return (f"rings={self.n_rings}  mean size={self.mean_size:.1f}\n"
                f"purity={self.mean_purity:.3f} (weighted {self.weighted_purity:.3f})  "
                f"consolidation={self.consolidation_ratio:.1f} alerts/case  "
                f"coverage={self.coverage:.3f}")


def evaluate_rings(rings, y_true) -> RingMetrics:
    """Ring-level quality.

    Nobody in our literature review reports these, which is precisely why they
    are our most distinctive numbers.

    purity            — of the labelled members of a ring, what fraction are fraud
    consolidation     — how many transaction-level alerts collapse into one case
                        (an analyst's actual workload reduction)
    coverage          — fraction of all known fraud that lands inside some ring
    """
    y_true = np.asarray(y_true)
    if not rings:
        return RingMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    purities, sizes, weights, members = [], [], [], set()
    for r in rings:
        raw = r.node_ids if hasattr(r, "node_ids") else r["node_ids"]
        idx = np.asarray(raw, dtype=int)
        sizes.append(len(idx))
        members.update(idx.tolist())
        lab = y_true[idx]
        known = lab >= 0
        if known.sum():
            purities.append(float((lab[known] == 1).mean()))
            weights.append(int(known.sum()))

    total_fraud = int((y_true == 1).sum())
    caught = int((y_true[list(members)] == 1).sum()) if members else 0
    w = np.array(weights, dtype=float) if weights else np.array([0.0])

    return RingMetrics(
        n_rings=len(rings),
        mean_size=float(np.mean(sizes)),
        mean_purity=float(np.mean(purities)) if purities else 0.0,
        weighted_purity=float(np.average(purities, weights=w)) if purities else 0.0,
        consolidation_ratio=float(np.mean(sizes)),
        coverage=caught / total_fraud if total_fraud else 0.0)


def compare(results: dict, key: str = "pr_auc") -> str:
    """Rank named results and report lift over the best baseline.

    Relative lift over the strongest baseline is the honest comparison — an
    absolute PR-AUC number means nothing without knowing what the floor is.
    """
    rows = sorted(results.items(), key=lambda kv: -_get(kv[1], key))
    best_name, best = rows[0]
    w = max(len(n) for n, _ in rows) + 2
    lines = [f"{'model':<{w}}{key:>10}{'lift':>10}", "-" * (w + 20)]
    for name, m in rows:
        v = _get(m, key)
        lift = "" if name == best_name else f"{100*(v/_get(best, key)-1):+.1f}%"
        lines.append(f"{name:<{w}}{v:>10.4f}{lift:>10}")
    return "\n".join(lines)


def _get(m, key):
    return getattr(m, key) if hasattr(m, key) else m[key]
