"""Evaluate a trained GraphGuardModel: AUC, F1, PR-AUC, Precision@k on the test split,
plus ring purity (fraction of ring members that are actually fraud, among labeled ones).

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/best.pt --config configs/default.yaml
    python scripts/evaluate.py --synthetic   # smoke-test the metrics code with a fresh model
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.models.graphguard import GraphGuardModel
from src.ring.extract import extract_rings
from src.utils.config import get_logger, load_config, set_seed

logger = get_logger(__name__)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scored nodes that are actually fraud (label==1).
    The metric fraud teams actually act on: an analyst only ever reviews the top k alerts,
    so overall AUC can look fine while precision among the alerts an analyst would see is bad."""
    if k <= 0 or len(scores) == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k_idx].sum() / k)


def evaluate_classification(y_true: np.ndarray, y_score: np.ndarray, k_values: tuple[int, ...] = (10, 50, 100)) -> dict:
    """AUC/F1/PR-AUC/Precision@k on labeled nodes only (-1 already excluded by the caller)."""
    metrics = {
        "auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, (y_score >= 0.5).astype(int)),
    }
    for k in k_values:
        if k <= len(y_true):
            metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
    return metrics


def ring_purity(rings: list, y_true: np.ndarray) -> float:
    """Mean, across rings, of (fraction of a ring's *labeled* members that are fraud).
    Rings with zero labeled members are skipped -- purity is undefined for them, not 0."""
    purities = []
    for ring in rings:
        labels = np.array([y_true[n] for n in ring.node_ids if y_true[n] != -1])
        if len(labels) == 0:
            continue
        purities.append((labels == 1).mean())
    return float(np.mean(purities)) if purities else float("nan")


def evaluate(model: GraphGuardModel, data, ring_cfg: dict) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(data)

    test_mask = data.test_mask.numpy()
    y_true_all = data.y.numpy()

    labeled_test = test_mask & (y_true_all != -1)
    y_true = y_true_all[labeled_test]
    fraud_score = out["fraud_score"].numpy()[labeled_test]

    classification_metrics = evaluate_classification(y_true, fraud_score)

    rings = extract_rings(
        data, out["fraud_score"], threshold=ring_cfg["score_threshold"], min_size=ring_cfg["min_size"]
    )
    purity = ring_purity(rings, y_true_all)

    results = {**classification_metrics, "ring_purity": purity, "n_rings": len(rings)}
    logger.info("Evaluation results: %s", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    if args.synthetic:
        from scripts.train import make_synthetic_dataset, train

        data = make_synthetic_dataset()
        logger.warning("--synthetic: training a fresh throwaway model to smoke-test the metrics code.")
        model = train(cfg, data, checkpoint_dir=__import__("pathlib").Path("results/checkpoints"))
    else:
        from scripts.train import load_real_or_synthetic

        data = load_real_or_synthetic(cfg["model"]["dataset"], cfg)
        model = GraphGuardModel.load(args.checkpoint)

    evaluate(model, data, cfg["ring"])


if __name__ == "__main__":
    main()
