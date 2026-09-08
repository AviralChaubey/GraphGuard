"""Shared synthetic fixtures. Nothing here touches a real dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.data.base import GraphBundle


@pytest.fixture
def toy_bundle() -> GraphBundle:
    """Small homogeneous graph with learnable, time-dependent signal."""
    rng = np.random.default_rng(0)
    n = 600
    t = rng.integers(0, 20, n)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0 * X[:, 1] - 2.0)))
    y = (rng.random(n) < p).astype(int)
    y[rng.random(n) < 0.2] = -1                      # unknowns
    b = GraphBundle(name="toy", target_node_type="account")
    b.nodes["account"] = pd.DataFrame({"node_id": [f"a{i}" for i in range(n)]})
    b.edges[("account", "sent_to", "account")] = pd.DataFrame(
        {"src": rng.integers(0, n, 900), "dst": rng.integers(0, n, 900)})
    b.features["account"] = X
    b.labels["account"] = y
    b.time["account"] = t
    return b


@pytest.fixture
def hetero_bundle(tmp_path) -> GraphBundle:
    """IEEE-CIS-shaped bundle containing a planted ring: one device on many cards."""
    from src.data.ieee_cis import load_ieee_cis
    rng = np.random.default_rng(1)
    n = 900
    dev = list(rng.choice([f"D{i}" for i in range(60)], n))
    for i in range(80):
        dev[i] = "RING_DEVICE"                       # the planted ring
    pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": rng.binomial(1, 0.04, n),
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
    }).to_csv(tmp_path / "train_transaction.csv", index=False)
    return load_ieee_cis(root=tmp_path)