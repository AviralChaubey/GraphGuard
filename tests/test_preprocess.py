"""Smoke tests that run without any real dataset downloaded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.data.base import GraphBundle, random_split, temporal_split
from src.data.ieee_cis import _entity_key


def test_temporal_split_has_no_overlap():
    t = np.arange(100)
    tr, te = temporal_split(t, 0.7)
    assert not (tr & te).any()
    assert tr.sum() + te.sum() == 100
    assert t[tr].max() < t[te].min()      # train is strictly in the past


def test_random_split_partitions():
    tr, te = random_split(50, 0.8, seed=1)
    assert tr.sum() == 40 and te.sum() == 10
    assert not (tr & te).any()


def test_entity_key_joins_and_nulls():
    df = pd.DataFrame({"a": ["x", None, "z"], "b": ["1", None, None]})
    k = _entity_key(df, ["a", "b"])
    assert k.iloc[0] == "x|1"
    assert pd.isna(k.iloc[1])             # all components missing -> NA
    assert k.iloc[2] == "z|?"


def test_bundle_roundtrip(tmp_path):
    b = GraphBundle(name="toy", target_node_type="account")
    b.nodes["account"] = pd.DataFrame({"node_id": ["a", "b", "c"]})
    b.edges[("account", "sent_to", "account")] = pd.DataFrame({"src": [0, 1], "dst": [1, 2]})
    b.features["account"] = np.random.rand(3, 4).astype(np.float32)
    b.labels["account"] = np.array([1, 0, -1])
    b.time["account"] = np.array([0, 1, 2])
    b.save(tmp_path)
    assert (tmp_path / "arrays.npz").exists()
    assert "fraud=1" in b.summary()
