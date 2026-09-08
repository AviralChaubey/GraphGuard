"""Baselines that answer: does the graph help at all?

Two tiers, and the distinction matters in the write-up:

  NON-GRAPH   logistic regression, gradient boosting, majority — node features
              only, no edges. If our GNN cannot beat these, the graph adds
              nothing and the project's premise fails.

  GRAPH-LITE  the same tabular models trained on features PLUS hand-computed
              network statistics (degree, clustering, PageRank). This is the
              honest baseline: it separates "graph information helps" from
              "our particular GNN helps". Beating plain XGBoost is easy;
              beating XGBoost-with-degree-features is the real test.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.data.base import GraphBundle
from src.eval.metrics import DetectionMetrics, evaluate_detection
from src.utils.config import get_logger

log = get_logger(__name__)


def _fit_predict(model, X, y, train, test) -> np.ndarray:
    """Fit on train, return scores for ALL rows (metrics apply the test mask)."""
    tr = train & (y >= 0)
    if tr.sum() == 0 or y[tr].sum() == 0:
        raise ValueError("training split contains no positive labels")
    model.fit(X[tr], y[tr])
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def run_baselines(bundle: GraphBundle, train_mask, test_mask,
                  node_type: str | None = None,
                  include_network_features: bool = True) -> dict:
    """Returns {name: DetectionMetrics}."""
    nt = node_type or bundle.target_node_type
    X = bundle.features[nt]
    y = bundle.labels[nt]
    train = np.asarray(train_mask, dtype=bool)
    test = np.asarray(test_mask, dtype=bool)

    # class_weight="balanced" everywhere: without it, a 2% positive rate makes
    # the majority class dominate the loss and every model predicts "licit".
    specs = {
        "majority (never fraud)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "gradient boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, class_weight="balanced",
            early_stopping=True, random_state=42),
    }

    results: dict[str, DetectionMetrics] = {}
    for name, model in specs.items():
        try:
            scores = _fit_predict(model, X, y, train, test)
            results[name] = evaluate_detection(y, scores, mask=test)
            log.info("%-24s PR-AUC %.4f", name, results[name].pr_auc)
        except Exception as e:                        # noqa: BLE001
            log.warning("baseline '%s' failed: %s", name, e)

    if include_network_features:
        Xn = _with_network_features(bundle, nt)
        if Xn is not None:
            model = HistGradientBoostingClassifier(
                max_iter=200, class_weight="balanced",
                early_stopping=True, random_state=42)
            try:
                scores = _fit_predict(model, Xn, y, train, test)
                results["gradient boosting + network feats"] = evaluate_detection(
                    y, scores, mask=test)
                log.info("%-24s PR-AUC %.4f", "GB + network",
                         results["gradient boosting + network feats"].pr_auc)
            except Exception as e:                    # noqa: BLE001
                log.warning("graph-lite baseline failed: %s", e)

    return results


def _with_network_features(bundle: GraphBundle, nt: str):
    """Append degree / clustering / PageRank without mutating the bundle."""
    import copy

    from src.graph.build import add_network_features

    rel = next((r for r in bundle.edges if r[0] == nt and r[2] == nt), None)
    if rel is None:
        log.info("no %s-%s edges; skipping graph-lite baseline", nt, nt)
        return None
    tmp = copy.copy(bundle)
    tmp.features = dict(bundle.features)
    add_network_features(tmp, nt)
    return tmp.features[nt]
