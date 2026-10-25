#!/usr/bin/env python
"""Evaluation, baselines, ring quality and the leakage experiment.

MERGED FILE — workstreams A and B both wrote a scripts/evaluate.py.
This is the single entry point; nothing was dropped from either version.

    # baselines only — runs today, no trained model needed
    python -m scripts.evaluate --baselines

    # a trained checkpoint: detection metrics + ring quality
    python -m scripts.evaluate --checkpoint results/checkpoints/best.pt

    # everything, compared side by side
    python -m scripts.evaluate --baselines --checkpoint results/checkpoints/best.pt

    # the leakage experiment (temporal vs random split)
    python -m scripts.evaluate --leakage

    # smoke-test the metrics code on a throwaway model
    python -m scripts.evaluate --synthetic

--dataset is optional and defaults to `data.dataset` in the config.
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
from src.eval import (compare, evaluate_detection, evaluate_rings,
                      run_baselines)
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


def _score_and_rings(model, data, ring_cfg: dict):
    """Run the model and extract rings.

    Touches only the frozen contract — `forward()` returning `fraud_score`, and
    `extract_rings()`. B can change everything inside the model without
    breaking evaluation.

    Ring extraction is homogeneous-graph-only right now (see the module
    docstring in src/models/graphguard.py for the hetero gap -- that's a
    separate, not-yet-done piece of work, not a bug in the model itself).
    A HeteroData input must still produce real detection metrics, so a ring
    extraction failure here is caught and reported as "no rings" rather than
    letting it take down the fraud_score/PR-AUC numbers with it.
    """
    import torch
    from torch_geometric.data import HeteroData

    from src.ring.extract import extract_rings

    model.eval()
    with torch.no_grad():
        out = model(data)

    fraud_score = out["fraud_score"].cpu()
    if isinstance(data, HeteroData):
        log.warning(
            "ring extraction does not support heterogeneous graphs yet "
            "(known gap, see src/models/graphguard.py) -- skipping rings, "
            "detection metrics are still computed from fraud_score."
        )
        return fraud_score.numpy(), []
    rings = extract_rings(
        data, fraud_score,
        threshold=ring_cfg["score_threshold"],
        min_size=ring_cfg["min_size"],
    )
    return fraud_score.numpy(), rings


def _target_labels(data, bundle):
    """y and test mask for the target node type, for Data or HeteroData."""
    from torch_geometric.data import HeteroData

    if isinstance(data, HeteroData):
        nt = getattr(data, "target_node_type", bundle.target_node_type)
        return data[nt].y.numpy(), data[nt].test_mask.numpy()
    return data.y.numpy(), data.test_mask.numpy()


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
    p.add_argument("--synthetic", action="store_true",
                   help="train a throwaway model on synthetic data to smoke-test "
                        "the metrics code — never a real result")
    p.add_argument("--tag", default="", help="suffix for the output filename")
    args = p.parse_args()

    if not (args.baselines or args.leakage or args.checkpoint or args.synthetic):
        p.error("nothing to do — pass --baselines, --leakage, --checkpoint "
                "and/or --synthetic")

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    results, blocks = {}, []
    ring_metrics = None

    # ---------------- synthetic smoke test -------------------------------
    if args.synthetic:
        from scripts.train import make_synthetic_dataset, train

        log.warning("--synthetic: throwaway model on generated data. "
                    "These numbers are a code smoke test, NOT a result. "
                    "Do not put them in the report.")
        data = make_synthetic_dataset()
        model = train(cfg, data, checkpoint_dir=PROJECT_ROOT / "results" / "checkpoints")
        scores, rings = _score_and_rings(model, data, cfg["ring"])
        y = data.y.numpy()
        results["GraphGuard (synthetic)"] = evaluate_detection(
            y, scores, mask=data.test_mask.numpy())
        ring_metrics = evaluate_rings(rings, y)

    # ---------------- real dataset ---------------------------------------
    else:
        dataset = args.dataset or cfg["data"]["dataset"]
        log.info("dataset: %s", dataset)

        kwargs = {"nrows": args.nrows} if args.nrows and dataset != "elliptic" else {}
        bundle = load_dataset(dataset, **kwargs)
        train_mask, test_mask = _load_splits(
            dataset, bundle, cfg["data"].get("train_ratio", 0.7))

        y = bundle.labels[bundle.target_node_type]
        log.info("test set: %d nodes, %d labelled positives",
                 int(test_mask.sum()), int((y[test_mask] == 1).sum()))

        if args.baselines:
            results.update(run_baselines(bundle, train_mask, test_mask))

        if args.checkpoint:
            try:
                from src.graph import to_pyg
                from src.models.graphguard import GraphGuardModel

                data = to_pyg(bundle, train_mask=train_mask, test_mask=test_mask)
                model = GraphGuardModel.load(args.checkpoint)
                scores, rings = _score_and_rings(model, data, cfg["ring"])
                y_eval, te = _target_labels(data, bundle)
                results["GraphGuard"] = evaluate_detection(y_eval, scores, mask=te)
                ring_metrics = evaluate_rings(rings, y_eval)
            except Exception as e:                    # noqa: BLE001
                log.error("could not evaluate checkpoint: %s", e)

        if args.leakage:
            blocks.append(format_report(run_leakage_experiment(bundle)))

    # ---------------- report ---------------------------------------------
    out_blocks = []
    if results:
        out_blocks.append("Detection results (test split)\n" + "=" * 48)
        for name, m in results.items():
            out_blocks.append(f"\n[{name}]\n{m}")
        out_blocks.append("\n\n" + compare(results, "pr_auc"))

    if ring_metrics is not None:
        out_blocks.append("\n\nRing quality\n" + "=" * 48 + f"\n{ring_metrics}")

    out_blocks.extend("\n\n" + b for b in blocks)
    report = "\n".join(out_blocks)
    print("\n" + report)

    out = PROJECT_ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    ds = "synthetic" if args.synthetic else (args.dataset or cfg["data"]["dataset"])
    stem = f"{ds}_eval{('_' + args.tag) if args.tag else ''}"
    (out / f"{stem}.txt").write_text(report + "\n")
    if results:
        payload = {k: v.to_dict() for k, v in results.items()}
        if ring_metrics is not None:
            payload["_rings"] = ring_metrics.to_dict()
        (out / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    log.info("written -> %s", out / f"{stem}.txt")


if __name__ == "__main__":
    main()
