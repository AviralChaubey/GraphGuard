"""Fallback test runner for environments without pytest installed.

Normally you just run `pytest tests/ -q`. This file exists so the suite can be
executed anywhere; it wires up the same fixtures conftest.py provides.
"""
import inspect
import logging
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

try:                                    # pytest is the normal runner
    import pytest                       # noqa: F401
except ImportError:                     # fallback: minimal shim so imports work
    import contextlib
    import types

    shim = types.ModuleType("pytest")
    shim.fixture = lambda f=None, **kw: (f if f else (lambda g: g))

    @contextlib.contextmanager
    def _raises(exc, match=None):
        import re
        try:
            yield
        except exc as e:
            if match and not re.search(match, str(e)):
                raise AssertionError(
                    f"{exc.__name__} raised but message {str(e)!r} "
                    f"does not match {match!r}")
        else:
            raise AssertionError(f"{exc.__name__} was not raised")

    shim.raises = _raises
    shim.mark = types.SimpleNamespace(
        parametrize=lambda *a, **k: (lambda f: f),
        skip=lambda *a, **k: (lambda f: f))
    sys.modules["pytest"] = shim

from src.data.base import GraphBundle


def toy_bundle():
    rng = np.random.default_rng(0)
    n = 600
    t = rng.integers(0, 20, n)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0 * X[:, 1] - 2.0)))
    y = (rng.random(n) < p).astype(int)
    y[rng.random(n) < 0.2] = -1
    b = GraphBundle(name="toy", target_node_type="account")
    b.nodes["account"] = pd.DataFrame({"node_id": [f"a{i}" for i in range(n)]})
    b.edges[("account", "sent_to", "account")] = pd.DataFrame(
        {"src": rng.integers(0, n, 900), "dst": rng.integers(0, n, 900)})
    b.features["account"] = X
    b.labels["account"] = y
    b.time["account"] = t
    return b


def hetero_bundle():
    from src.data.ieee_cis import load_ieee_cis
    rng = np.random.default_rng(1)
    n = 900
    dev = list(rng.choice([f"D{i}" for i in range(60)], n))
    for i in range(80):
        dev[i] = "RING_DEVICE"
    tp = pathlib.Path(tempfile.mkdtemp())
    pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": rng.binomial(1, .04, n),
        "TransactionDT": rng.integers(86400, 86400 * 60, n),
        "TransactionAmt": rng.gamma(2, 50, n),
        "card1": rng.integers(1000, 1100, n),
        "addr1": rng.integers(200, 220, n),
        "P_emaildomain": rng.choice(["a.com", "b.com", None], n),
        "DeviceInfo": dev,
        # DeviceType must be a function of the device: the entity key is
        # DeviceInfo|DeviceType|..., so a random type would split one physical
        # device into two entity nodes and halve its fan-out.
        "DeviceType": ["mobile" if d == "RING_DEVICE" else "desktop" for d in dev],
        "C1": rng.random(n), "D1": rng.random(n),
    }).to_csv(tp / "train_transaction.csv", index=False)
    return load_ieee_cis(root=tp)


FIXTURES = {"toy_bundle": toy_bundle, "hetero_bundle": hetero_bundle}
MODULES = ["tests.test_eval", "tests.test_graph", "tests.test_eda",
           "tests.test_paysim_features", "tests.test_preprocess"]


def main() -> int:
    logging.disable(logging.INFO)
    passed = failed = 0
    failures = []
    for mn in MODULES:
        try:
            mod = __import__(mn, fromlist=["*"])
        except Exception as e:                       # noqa: BLE001
            failures.append(f"{mn}: import error: {e}")
            failed += 1
            continue
        for name, fn in vars(mod).items():
            if not (name.startswith("test_") and callable(fn)):
                continue
            kw = {}
            for p in inspect.signature(fn).parameters:
                if p == "tmp_path":
                    kw[p] = pathlib.Path(tempfile.mkdtemp())
                elif p in FIXTURES:
                    kw[p] = FIXTURES[p]()
            try:
                fn(**kw)
                passed += 1
            except Exception as e:                   # noqa: BLE001
                failed += 1
                failures.append(f"{mn}.{name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    for f in failures:
        print("  FAIL", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
