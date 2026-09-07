#!/usr/bin/env python
"""Evaluation, baselines and the leakage experiment.

    # baselines only — runs today, no trained model needed
    python scripts/evaluate.py --dataset elliptic --baselines

    # the leakage experiment
    python scripts/evaluate.py --dataset elliptic --leakage

    # score a trained checkpoint from B, and compare it against the baselines
    python scripts/evaluate.py --dataset elliptic --baselines \
        --checkpoint results/checkpoints/best.pt

Results are printed and written to results/metrics/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_dataset, temporal_split
from src.eval import compare, evaluate_detection, run_baselines
from src.eval.leakage import format_report, run_leakage_experiment
from src.utils.config import PROJECT_ROOT, get_logger, load_config, set_seed

log = get_logger("evaluate")


def _load_splits(dataset: str, bundle, train_ratio: float):
    """Prefer the saved split so every experiment uses identical partitions."""
    f = PROJECT_ROOT / "data" / "processed" / dataset / "splits.npz"
    if f.exists():
        z = np.load(f)
        log.info("using saved splits from %s", f.name)
        return z["train"], z["test"]
    log.warning("no splits.npz — recomputing (run preprocess.py first)")
    return temporal_split(bundle.time[bundle.target_node_type], train_ratio)


def _score_checkpoint(path: str, bundle):
    """Score a checkpoint from workstream B through the frozen contract.

    We only ever touch `forward()['fraud_score']` — B is free to change
    everything inside the model without breaking evaluation.
    """
    import torch

    from src.graph import to_pyg
    from src.models.graphguard import GraphGuardModel

    model = GraphGuardModel.load(path)
    model.eval()
    with torch.no_grad():
        out = model(to_pyg(bundle))
    return out["fraud_score"].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None,
                   choices=["elliptic", "ieee_cis", "paysim"],
                   help="defaults to data.dataset in the config file")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--baselines", action="store_true", help="run baseline models")
    p.add_argument("--leakage", action="store_true", help="temporal vs random split")
    p.add_argument("--checkpoint", default=None, help="trained model to evaluate")
    p.add_argument("--tag", default="", help="suffix for the output filename")
    args = p.parse_args()

    if not (args.baselines or args.leakage or args.checkpoint):
        p.error("nothing to do — pass --baselines, --leakage and/or --checkpoint")

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    dataset = args.dataset or cfg["data"]["dataset"]
    log.info("dataset: %s", dataset)

    kwargs = {"nrows": args.nrows} if args.nrows and dataset != "elliptic" else {}
    bundle = load_dataset(dataset, **kwargs)
    train, test = _load_splits(dataset, bundle, cfg["data"].get("train_ratio", 0.7))

    y = bundle.labels[bundle.target_node_type]
    log.info("test set: %d nodes, %d labelled positives",
             int(test.sum()), int((y[test] == 1).sum()))

    results, blocks = {}, []

    if args.baselines:
        results.update(run_baselines(bundle, train, test))

    if args.checkpoint:
        try:
            scores = _score_checkpoint(args.checkpoint, bundle)
            results["GraphGuard"] = evaluate_detection(y, scores, mask=test)
        except Exception as e:                        # noqa: BLE001
            log.error("could not evaluate checkpoint: %s", e)

    if results:
        blocks.append("Detection results (test split)\n" + "=" * 48)
        for name, m in results.items():
            blocks.append(f"\n[{name}]\n{m}")
        blocks.append("\n\n" + compare(results, "pr_auc"))

    if args.leakage:
        blocks.append("\n\n" + format_report(run_leakage_experiment(bundle)))

    report = "\n".join(blocks)
    print("\n" + report)

    out = PROJECT_ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_eval{('_' + args.tag) if args.tag else ''}"
    (out / f"{stem}.txt").write_text(report + "\n")
    if results:
        (out / f"{stem}.json").write_text(json.dumps(
            {k: v.to_dict() for k, v in results.items()}, indent=2))
    log.info("written -> %s", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
