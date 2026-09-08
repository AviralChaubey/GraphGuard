import matplotlib
matplotlib.use("Agg")

from src.data.eda import (degree_distribution, fan_out_histogram,
                          label_coverage, run_all)


def test_fanout_finds_the_planted_ring(hetero_bundle, tmp_path):
    """The fixture plants one device on 80 transactions; the histogram must
    surface it. This is the figure our ring claim rests on."""
    res = fan_out_histogram(hetero_bundle, out_dir=tmp_path)
    assert res is not None
    dev = res["stats"]["device"]
    assert dev["max"] >= 80
    assert dev["max"] > dev["median"] * 5        # a genuine outlier, not the bulk
    assert res["path"].exists()


def test_all_figures_render(hetero_bundle, tmp_path):
    res = run_all(hetero_bundle, out_dir=tmp_path)
    written = [v["path"] for v in res.values()
               if isinstance(v, dict) and "path" in v]
    assert written and all(p.exists() and p.stat().st_size > 1000 for p in written)


def test_fraud_concentration_ranks_by_wilson_bound(hetero_bundle):
    """Concentration, not popularity, is the ring signal."""
    from src.data.eda import fraud_concentration
    con = fraud_concentration(hetero_bundle, min_txns=2)
    assert 0 < con["base_rate"] < 1
    for rows in con["entities"].values():
        lbs = [r["wilson_lb"] for r in rows]
        assert lbs == sorted(lbs, reverse=True)
        for r in rows:
            assert 0 <= r["fraud_rate"] <= 1
            assert r["wilson_lb"] <= r["fraud_rate"]   # a lower bound
            assert r["n_fraud"] <= r["n_txns"]


def test_wilson_prefers_tight_ring_over_big_bucket():
    """The ordering the whole ring claim depends on."""
    import numpy as np

    from src.data.eda import _wilson_lower_bound

    ring = _wilson_lower_bound(np.array([40]), np.array([45]))[0]
    bucket = _wilson_lower_bound(np.array([422]), np.array([4314]))[0]
    fluke = _wilson_lower_bound(np.array([2]), np.array([2]))[0]
    assert ring > fluke > bucket


def test_degree_plot_skipped_without_homogeneous_edges(hetero_bundle, tmp_path):
    """IEEE-CIS has no transaction-transaction edges; skip, don't crash."""
    assert degree_distribution(hetero_bundle, out_dir=tmp_path) is None


def test_label_coverage_counts(hetero_bundle, tmp_path):
    res = label_coverage(hetero_bundle, out_dir=tmp_path)
    rows = res["rows"]["transaction"]
    assert rows["fraud"] + rows["licit"] == len(hetero_bundle.nodes["transaction"])
    # entity nodes are unlabelled conduits, never classified
    assert res["rows"]["device"]["unknown"] == len(hetero_bundle.nodes["device"])
