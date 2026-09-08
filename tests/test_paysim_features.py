import numpy as np
import pandas as pd

from src.data.paysim import load_paysim


def _write(tmp_path, n=1200, mule_frac=0.08, seed=0):
    rng = np.random.default_rng(seed)
    amt = rng.gamma(2, 4000, n)
    old_o = amt * rng.uniform(1.1, 3.0, n)
    old_d = rng.gamma(2, 2500, n)
    new_d = old_d + amt
    mule = rng.random(n) < mule_frac
    new_d[mule] = old_d[mule]                 # money passes straight through
    pd.DataFrame({
        "step": rng.integers(1, 745, n),
        "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n),
        "amount": amt,
        "nameOrig": [f"C{i % 200}" for i in range(n)],
        "oldbalanceOrg": old_o, "newbalanceOrig": np.maximum(old_o - amt, 0),
        "nameDest": [(f"M{i % 20}" if i % 4 == 0 else f"C{(i * 3) % 200}")
                     for i in range(n)],
        "oldbalanceDest": old_d, "newbalanceDest": new_d,
        "isFraud": rng.binomial(1, 0.03, n), "isFlaggedFraud": 0,
    }).to_csv(tmp_path / "paysim.csv", index=False)
    return tmp_path


def test_balance_features_present_and_finite(tmp_path):
    b = load_paysim(root=_write(tmp_path))
    X = b.features["account"]
    assert X.shape[1] >= 18            # aggregates + balance features
    assert np.isfinite(X).all()        # no NaN/inf from divisions


def test_merchant_flag_marks_merchants(tmp_path):
    b = load_paysim(root=_write(tmp_path))
    ids = b.nodes["account"]["node_id"].to_numpy()
    is_merchant = b.features["account"][:, -1]
    m = np.array([str(v).startswith("M") for v in ids])
    assert is_merchant[m].sum() > 0
    assert is_merchant[~m].sum() == 0   # customers never flagged


def test_loader_survives_missing_balance_columns(tmp_path):
    """Balance columns absent -> warn and continue, never crash."""
    _write(tmp_path)
    df = pd.read_csv(tmp_path / "paysim.csv").drop(
        columns=["oldbalanceOrg", "newbalanceOrig",
                 "oldbalanceDest", "newbalanceDest"])
    df.to_csv(tmp_path / "paysim.csv", index=False)
    b = load_paysim(root=tmp_path)
    assert b.features["account"].shape[1] >= 8
