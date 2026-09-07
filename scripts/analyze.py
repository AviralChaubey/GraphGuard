#!/usr/bin/env python
"""Generate EDA figures and a findings report for one dataset.

    python scripts/analyze.py --dataset ieee_cis --nrows 200000
    python scripts/analyze.py --dataset elliptic

Figures land in results/figures/; a plain-text findings summary is printed and
written to results/metrics/<dataset>_eda.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_dataset
from src.data.eda import run_all
from src.utils.config import PROJECT_ROOT, get_logger, load_config, set_seed

log = get_logger("analyze")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["elliptic", "ieee_cis", "paysim"])
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    kwargs = {"nrows": args.nrows} if args.nrows and args.dataset != "elliptic" else {}
    bundle = load_dataset(args.dataset, **kwargs)
    res = run_all(bundle)

    lines = [f"EDA — {bundle.name}", "=" * 40, "", bundle.summary(), ""]

    fo = res.get("fan_out")
    if fo:
        lines.append("Entity fan-out (evidence that rings exist in the data):")
        for ent, s in fo["stats"].items():
            lines.append(f"  {ent:<9} {s['n_entities']:>8,} entities | "
                         f"median {s['median']:>6.1f} | p99 {s['p99']:>8.1f} | "
                         f"MAX {s['max']:>8,}")
        top = max(fo["stats"].items(), key=lambda kv: kv[1]["max"])
        lines += ["", f"  >> Headline: one {top[0]} is linked to {top[1]['max']:,} "
                      f"transactions, against a median of {top[1]['median']:.0f}.",
                  "     A per-transaction model cannot see that concentration.", ""]

    dg = res.get("degree")
    if dg:
        lines += [f"Degree: max {dg['max_degree']:,}, mean {dg['mean_degree']:.2f}, "
                  f"isolated {dg['isolated']:,}", ""]

    ft = res.get("fraud_over_time")
    if ft:
        lines += [f"Fraud rate (labelled only): {ft['overall_rate']:.4%}",
                  f"Temporal split cutoff at time-step {ft['cutoff']:.0f}", ""]

    lc = res.get("label_coverage")
    if lc:
        lines.append("Label coverage:")
        for nt, c in lc["rows"].items():
            tot = sum(c.values())
            lines.append(f"  {nt:<12} fraud {c['fraud']:>7,} | licit {c['licit']:>8,} | "
                         f"unknown {c['unknown']:>8,}  ({c['unknown']/tot:.0%} unlabelled)")
        lines.append("")

    lines.append("Figures written to results/figures/")
    report = "\n".join(lines)
    print("\n" + report)

    out = PROJECT_ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.dataset}_eda.txt").write_text(report + "\n")
    (out / f"{args.dataset}_eda.json").write_text(json.dumps(
        {k: ({kk: vv for kk, vv in v.items() if kk != "path"} if isinstance(v, dict) else v)
         for k, v in res.items() if v}, indent=2, default=str))
    log.info("report -> %s", out / f"{args.dataset}_eda.txt")


if __name__ == "__main__":
    main()
