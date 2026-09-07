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
        lines.append("Entity fan-out:")
        lines.append(f"  {'entity':<9} {'distinct':>9} {'median':>8} {'p99':>10} "
                     f"{'max':>10}  kind")
        for ent, st in fo["stats"].items():
            kind = "identity" if st["is_identity"] else "CATEGORY (not a ring signal)"
            lines.append(f"  {ent:<9} {st['n_entities']:>9,} {st['median']:>8.1f} "
                         f"{st['p99']:>10.1f} {st['max']:>10,}  {kind}")

        lines.append("")
        lines.append("  Largest hubs (check these are real identities, not '?'-filled")
        lines.append("  keys — a key whose components were all missing would look like")
        lines.append("  a huge ring but is a data artifact):")
        for ent, st in fo["stats"].items():
            for name, cnt in st.get("top_hubs", [])[:3]:
                lines.append(f"    {ent:<8} {cnt:>8,}  {name[:70]}")

        ident = {k: v for k, v in fo["stats"].items() if v["is_identity"]}
        cats = [k for k, v in fo["stats"].items() if not v["is_identity"]]

        if cats:
            lines += ["",
                      f"  Category-like entity types: {', '.join(cats)}.",
                      "  A handful of distinct values cover nearly every transaction, so",
                      "  sharing one is not evidence of anything. Their fan-out is",
                      "  excluded from the ring claim below — consider modelling them as",
                      "  transaction features rather than node types."]

        lines += ["",
                  "  NOTE: fan-out alone is NOT ring evidence. It measures how popular",
                  "  an entity is, and the most popular values are common browser",
                  "  configurations. See fraud concentration below for the real signal.",
                  ""]

    con = res.get("concentration")
    if con and con["entities"]:
        lines.append("Fraud concentration — THE RING EVIDENCE")
        lines.append(f"  Base fraud rate: {con['base_rate']:.2%}")
        lines.append("  Entities whose transactions are disproportionately fraudulent:")
        lines.append("  Ranked by Wilson 95% lower bound — demands a high rate AND")
        lines.append("  enough transactions to believe it, so neither big buckets nor")
        lines.append("  2-transaction flukes can top the list.")
        lines.append(f"    {'entity':<8} {'txns':>6} {'fraud':>6} {'rate':>6} "
                     f"{'lift':>6} {'LB':>6}  key")
        best = None
        for ent, rows in con["entities"].items():
            for r in rows[:5]:
                lines.append(f"    {ent:<8} {r['n_txns']:>6,} {r['n_fraud']:>6,} "
                             f"{r['fraud_rate']:>5.0%} {r['lift']:>5.0f}x "
                             f"{r['wilson_lb']:>5.0%}  {r['node_id'][:52]}")
                if best is None or r["wilson_lb"] > best[1]["wilson_lb"]:
                    best = (ent, r)
        if best:
            ent, r = best
            lines += ["",
                      f"  >> Headline: one {ent} carries {r['n_txns']} transactions, "
                      f"{r['n_fraud']} of them",
                      f"     fraudulent — {r['fraud_rate']:.0%} against a "
                      f"{con['base_rate']:.1%} base rate ({r['lift']:.0f}x lift, "
                      f"95% LB {r['wilson_lb']:.0%}).",
                      "     Each of those transactions looks ordinary alone. The",
                      "     concentration is only visible in the graph.", ""]

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
