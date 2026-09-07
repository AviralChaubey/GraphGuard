"""Exploratory analysis figures for GraphBundle objects.

Every figure here answers a question we will be asked in review:

  fan_out_histogram   -> "what evidence do you have that rings exist?"
  degree_distribution -> "is this graph structurally realistic?"
  fraud_rate_over_time-> "is a temporal split fair, or does fraud drift?"
  label_coverage      -> "why do you need an unsupervised branch?"

Figures are written to results/figures/ as PNG at 200 dpi.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import GraphBundle
from src.utils.config import PROJECT_ROOT, get_logger

log = get_logger(__name__)

# Validated categorical slots 1-3 (blue / orange / aqua). Fixed order, never cycled.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e3e2de"

FIG_DIR = PROJECT_ROOT / "results" / "figures"

# Below this ratio of (distinct entities / edges), an entity type is a category
# rather than an identity, and its fan-out carries no ring information.
IDENTITY_RATIO_MIN = 0.01


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Recessive grid and axes; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=10, fontweight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    return ax


def _save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    log.info("wrote %s", path)
    return path


# ----------------------------------------------------------------------------
def fan_out_histogram(bundle: GraphBundle, out_dir=None):
    """Distribution of how many transactions each entity touches.

    THE key figure. A device used by 3 cards is a family sharing a tablet; a
    device used by 300 cards is an operation. This is empirical evidence that
    rings exist in the data, produced before any model is trained — so it is an
    observation about the dataset, not a claim about our method.
    """
    import matplotlib.pyplot as plt

    ent_rels = [(r, r[2]) for r in bundle.edges
                if r[0] == bundle.target_node_type and r[2] != bundle.target_node_type]
    if not ent_rels:
        log.warning("%s has no entity node types; skipping fan-out", bundle.name)
        return None

    n = len(ent_rels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    stats = {}

    for ax, (rel, ent) in zip(axes, ent_rels):
        counts = bundle.edges[rel].groupby("dst").size().to_numpy()
        n_ent, n_edge = len(counts), int(counts.sum())
        # An entity type is IDENTITY-LIKE if it has many distinct values
        # relative to the transactions it touches (a card, a device). It is
        # CATEGORY-LIKE if a handful of values cover everything
        # (P_emaildomain = "gmail.com"; addr1 = a region code).
        #
        # This distinction decides whether fan-out means anything. 40 cards on
        # one device is a ring. 74,699 transactions on gmail.com is Tuesday.
        identity_ratio = n_ent / max(n_edge, 1)
        is_identity = identity_ratio >= IDENTITY_RATIO_MIN
        stats[ent] = {"n_entities": n_ent, "max": int(counts.max()),
                      "median": float(np.median(counts)),
                      "p99": float(np.quantile(counts, 0.99)),
                      "identity_ratio": float(identity_ratio),
                      "is_identity": bool(is_identity)}
        if not is_identity:
            log.warning("entity '%s' looks CATEGORY-like (%d distinct values over "
                        "%d edges) — its fan-out is not a ring signal; consider "
                        "making it a transaction feature instead of a node type",
                        ent, n_ent, n_edge)

        # Name the biggest hubs. A high fan-out is only a ring if the key is a
        # real identity — a key full of '?' means many distinct entities
        # collapsed into one node because their components were missing, which
        # would be a data bug masquerading as a fraud ring.
        top_idx = np.argsort(-counts)[:5]
        names = bundle.nodes[ent]["node_id"].to_numpy()
        stats[ent]["top_hubs"] = [(str(names[i]), int(counts[i])) for i in top_idx]
        bins = np.logspace(0, np.log10(max(counts.max(), 2)), 30)
        colour = C1 if is_identity else MUTED
        ax.hist(counts, bins=bins, color=colour, edgecolor=SURFACE, linewidth=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        title = ent.capitalize() + ("" if is_identity else "  (category — not a ring signal)")
        _style(ax, title, "transactions per entity (log)",
               "entities (log)" if ax is axes[0] else "")
        # direct label: the number that makes the argument
        ax.axvline(counts.max(), color=C2 if is_identity else MUTED,
                   linewidth=2, alpha=.9)
        ax.annotate(f"max {counts.max():,}", xy=(counts.max(), ax.get_ylim()[1]),
                    xytext=(-6, -12), textcoords="offset points", ha="right",
                    color=C2 if is_identity else MUTED, fontsize=9.5,
                    fontweight="semibold")

    fig.suptitle(f"Entity fan-out — {bundle.name}", color=INK, fontsize=13,
                 x=0.005, ha="left", fontweight="semibold")
    fig.tight_layout()
    path = _save(fig, f"{bundle.name}_fanout", out_dir)
    plt.close(fig)
    return {"path": path, "stats": stats}


def _wilson_lower_bound(successes: np.ndarray, n: np.ndarray,
                        z: float = 1.96) -> np.ndarray:
    """Lower bound of the 95% Wilson confidence interval on a proportion.

    This is the ranking we need. Sorting by fraud COUNT rewards big entities
    (a 4,314-transaction browser bucket at 10% wins); sorting by fraud RATE
    rewards noise (2 transactions, both fraud = 100%). The Wilson lower bound
    asks "what is the *worst* the true rate plausibly is, given this much
    evidence?" — so it demands both a high rate and enough transactions to
    believe it. 18/20 beats 422/4314, which is the correct ordering for rings.
    """
    n = np.maximum(n, 1)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = phat + z ** 2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return (centre - margin) / denom


def fraud_concentration(bundle: GraphBundle, min_txns: int = 5,
                        top_n: int = 10) -> dict:
    """Entities whose transactions are disproportionately fraudulent.

    THIS is the ring evidence, not raw fan-out. Fan-out alone measures how
    *popular* an entity is — "Windows 10 + Chrome 63" tops that list and is not
    a ring. What identifies a ring is CONCENTRATION: an entity whose
    transactions are fraudulent far above the base rate.

    "One card key carries 41 transactions, 32 of them fraudulent — 77% against
    a 3% base rate" is a claim that survives someone asking what that entity is.
    "One device has 4,839 transactions" does not.

    Descriptive statistics over labelled data, used to characterise the dataset
    — not a model input, so there is no leakage concern.
    """
    y = bundle.labels[bundle.target_node_type]
    base = float((y == 1).sum() / max((y >= 0).sum(), 1))
    out = {"base_rate": base, "entities": {}}

    for (s_t, rel, d_t), e in bundle.edges.items():
        if s_t != bundle.target_node_type or d_t == s_t:
            continue
        lab = y[e["src"].to_numpy()]
        df = pd.DataFrame({"ent": e["dst"].to_numpy(), "y": lab})
        df = df[df["y"] >= 0]
        g = df.groupby("ent")["y"].agg(["size", "sum"])
        g = g[g["size"] >= min_txns]
        if g.empty:
            continue
        g["rate"] = g["sum"] / g["size"]
        g["wilson"] = _wilson_lower_bound(g["sum"].to_numpy(), g["size"].to_numpy())
        # Rank by the Wilson lower bound: high rate AND enough evidence.
        g = g.sort_values("wilson", ascending=False).head(top_n)

        names = bundle.nodes[d_t]["node_id"].to_numpy()
        out["entities"][d_t] = [
            {"node_id": str(names[i]), "n_txns": int(r["size"]),
             "n_fraud": int(r["sum"]), "fraud_rate": float(r["rate"]),
             "wilson_lb": float(r["wilson"]),
             "lift": float(r["rate"] / base) if base else 0.0}
            for i, r in g.iterrows()]
        top = out["entities"][d_t][0]
        log.info("entity '%s': top concentration %d/%d fraud (%.0f%%, %.0fx base, "
                 "Wilson LB %.0f%%)", d_t, top["n_fraud"], top["n_txns"],
                 100 * top["fraud_rate"], top["lift"], 100 * top["wilson_lb"])
    return out


def degree_distribution(bundle: GraphBundle, out_dir=None):
    """Log-log degree distribution of the target node type.

    Real transaction networks are heavy-tailed (a few hubs, a long tail). If
    this looks Gaussian, our graph construction is wrong.
    """
    import matplotlib.pyplot as plt

    nt = bundle.target_node_type
    rel = next((r for r in bundle.edges if r[0] == nt and r[2] == nt), None)
    if rel is None:
        log.warning("%s has no %s-%s edges; skipping degree plot", bundle.name, nt, nt)
        return None

    e = bundle.edges[rel]
    n = len(bundle.nodes[nt])
    deg = np.bincount(np.concatenate([e["src"].to_numpy(), e["dst"].to_numpy()]),
                      minlength=n)
    vals, counts = np.unique(deg[deg > 0], return_counts=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.9), facecolor=SURFACE)
    ax.scatter(vals, counts, s=18, color=C1, alpha=.75, edgecolor=SURFACE, linewidth=.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    _style(ax, f"Degree distribution — {bundle.name}", "degree (log)", "nodes (log)")
    ax.annotate(f"max degree {deg.max():,}\nisolated nodes {(deg == 0).sum():,}",
                xy=(.98, .95), xycoords="axes fraction", ha="right", va="top",
                color=INK2, fontsize=9.5)
    fig.tight_layout()
    path = _save(fig, f"{bundle.name}_degree", out_dir)
    plt.close(fig)
    return {"path": path, "max_degree": int(deg.max()),
            "isolated": int((deg == 0).sum()), "mean_degree": float(deg.mean())}


def fraud_rate_over_time(bundle: GraphBundle, out_dir=None):
    """Labelled fraud rate per time-step, with the temporal split marked.

    Justifies the temporal split — and if fraud rate drifts sharply, that is
    itself a finding worth reporting (concept drift).
    """
    import matplotlib.pyplot as plt

    nt = bundle.target_node_type
    y, t = bundle.labels[nt], bundle.time[nt]
    known = y >= 0
    if known.sum() == 0:
        return None

    df = pd.DataFrame({"t": t[known], "y": y[known]})
    g = df.groupby("t")["y"].agg(["mean", "size"])
    cutoff = np.quantile(t[known], 0.7)

    fig, ax = plt.subplots(figsize=(7.2, 3.9), facecolor=SURFACE)
    ax.plot(g.index, g["mean"] * 100, color=C1, linewidth=2)
    ax.axvline(cutoff, color=MUTED, linewidth=1.5, linestyle="--")
    ax.annotate("train  |  test", xy=(cutoff, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", color=INK2, fontsize=9.5)
    _style(ax, f"Fraud rate over time — {bundle.name}",
           "time step", "fraud rate (% of labelled)")
    fig.tight_layout()
    path = _save(fig, f"{bundle.name}_fraud_over_time", out_dir)
    plt.close(fig)
    return {"path": path, "overall_rate": float(y[known].mean()),
            "cutoff": float(cutoff)}


def label_coverage(bundle: GraphBundle, out_dir=None):
    """Fraud / licit / unknown split as one stacked bar per node type.

    This is the argument for the autoencoder branch: on Elliptic, three
    quarters of nodes carry no label at all.
    """
    import matplotlib.pyplot as plt

    rows = [(nt, int((v == 1).sum()), int((v == 0).sum()), int((v == -1).sum()))
            for nt, v in bundle.labels.items()]
    rows = [r for r in rows if sum(r[1:]) > 0]
    if not rows:
        return None

    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(rows) + 1.9), facecolor=SURFACE)
    left = np.zeros(len(rows))
    for j, (name, color) in enumerate([("fraud", C2), ("licit", C1), ("unknown", C3)]):
        vals = np.array([r[j + 1] for r in rows], dtype=float)
        tot = np.array([sum(r[1:]) for r in rows], dtype=float)
        pct = 100 * vals / tot
        ax.barh(labels, pct, left=left, color=color, label=name,
                height=.55, edgecolor=SURFACE, linewidth=2)   # 2px surface gap
        # direct labels — required: aqua sits below 3:1 on this surface
        for i, (p, l) in enumerate(zip(pct, left)):
            if p > 7:
                ax.text(l + p / 2, i, f"{p:.0f}%", ha="center", va="center",
                        color="white", fontsize=9.5, fontweight="semibold")
        left += pct

    _style(ax, f"Label coverage — {bundle.name}", "share of nodes (%)", "")
    ax.set_xlim(0, 100)
    leg = ax.legend(frameon=False, ncol=3, loc="lower left",
                    bbox_to_anchor=(0, 1.02), fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(INK2)
    fig.tight_layout()
    path = _save(fig, f"{bundle.name}_label_coverage", out_dir)
    plt.close(fig)
    return {"path": path,
            "rows": {r[0]: {"fraud": r[1], "licit": r[2], "unknown": r[3]} for r in rows}}


def run_all(bundle: GraphBundle, out_dir=None) -> dict:
    """Every figure for one dataset. Returns the numbers, not just the paths."""
    import matplotlib
    matplotlib.use("Agg")
    return {
        "fan_out": fan_out_histogram(bundle, out_dir),
        "concentration": fraud_concentration(bundle),
        "degree": degree_distribution(bundle, out_dir),
        "fraud_over_time": fraud_rate_over_time(bundle, out_dir),
        "label_coverage": label_coverage(bundle, out_dir),
    }
