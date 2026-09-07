"""Shared config, logging and seeding helpers.

NOTE (Workstream C): `src/utils/` is a *shared* folder per docs/CONTRACTS.md, not something
I own outright. This file only exists so my code has `load_config()` / `get_logger()` /
`set_seed()` to import while A and B haven't committed their own version yet. If a teammate's
`src/utils/config.py` already exists in the repo, drop this one and use theirs instead --
don't let two versions diverge silently.
"""
from __future__ import annotations

import logging
import os
import random
from functools import lru_cache
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = os.path.join("configs", "default.yaml")


@lru_cache(maxsize=None)
def load_config(path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load a YAML config file and return it as a plain dict.

    Cached so repeated calls (e.g. from every module at import time) don't re-read disk.
    Falls back to an empty dict if the file doesn't exist yet, so tests never need a real
    config on disk.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting. Use this instead of print()."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("GRAPHGUARD_LOG_LEVEL", "INFO"))
    return logger


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
