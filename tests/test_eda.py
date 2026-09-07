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
    written = [v["path"] for v in res.values() if v]
    assert written and all(p.exists() and p.stat().st_size > 1000 for p in written)


def test_degree_plot_skipped_without_homogeneous_edges(hetero_bundle, tmp_path):
    """IEEE-CIS has no transaction-transaction edges; skip, don't crash."""
    assert degree_distribution(hetero_bundle, out_dir=tmp_path) is None


def test_label_coverage_counts(hetero_bundle, tmp_path):
    res = label_coverage(hetero_bundle, out_dir=tmp_path)
    rows = res["rows"]["transaction"]
    assert rows["fraud"] + rows["licit"] == len(hetero_bundle.nodes["transaction"])
    # entity nodes are unlabelled conduits, never classified
    assert res["rows"]["device"]["unknown"] == len(hetero_bundle.nodes["device"])
