#!/usr/bin/env python
"""Preprocess a raw dataset into a GraphBundle saved under data/processed/.

Usage:
    python scripts/preprocess.py --dataset elliptic
    python scripts/preprocess.py --dataset ieee_cis --nrows 50000
    python scripts/preprocess.py --dataset paysim --network-features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import LOADERS, temporal_split
from src.graph import add_network_features, normalize_features
from src.utils.config import get_logger, load_config, processed_dir, set_seed

log = get_logger("preprocess")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["elliptic", "ieee_cis", "paysim"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--nrows", type=int, default=None,
                   help="row cap for a fast smoke run (ieee_cis / paysim)")
    p.add_argument("--network-features", action="store_true",
                   help="add degree/clustering/pagerank (slow on big graphs)")
    p.add_argument("--no-normalize", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    kwargs = {}
    if args.nrows and args.dataset in ("ieee_cis", "paysim"):
        kwargs["nrows"] = args.nrows

    log.info("loading %s", args.dataset)
    # Always re-parse the raw files here, never the data/processed/ cache --
    # this script's whole job is to (re)build that cache, so load_dataset()'s
    # cache-preferring shortcut would make a re-run of preprocess.py just
    # re-save its own stale output and never pick up a code or raw-data change.
    bundle = LOADERS[args.dataset](**kwargs)

    if args.network_features:
        bundle = add_network_features(bundle)

    nt = bundle.target_node_type
    t = bundle.time[nt]
    train_mask, test_mask = temporal_split(t, cfg["data"].get("train_ratio", 0.7))
    log.info("temporal split: %d train / %d test nodes",
             int(train_mask.sum()), int(test_mask.sum()))

    if not args.no_normalize:
        bundle = normalize_features(bundle, train_mask)

    out = processed_dir(args.dataset)
    bundle.save(out)
    np.savez_compressed(out / "splits.npz", train=train_mask, test=test_mask)

    print()
    print(bundle.summary())
    y = bundle.labels[nt]
    known = y >= 0
    if known.sum():
        print(f"\n  fraud rate (labelled only): {y[known].mean():.4%}")
        print(f"  imbalance ratio           : 1 : {int((y[known]==0).sum()/max((y[known]==1).sum(),1))}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
