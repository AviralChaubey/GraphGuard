"""Random but correctly-shaped stand-in for GraphGuardModel.

Contract (docs/CONTRACTS.md, Contract 2): C builds src/explain/ and src/api/ against this
today and swaps in the real GraphGuardModel later without changing a single line on their
side, because the output dict has identical keys, shapes, and value ranges either way.

Deliberately has NO dependency on src.data or src.graph -- A's loaders aren't ready yet,
and this must work off nothing but a PyG-shaped `data` object (or anything with the same
two attributes: num_nodes and edge_index).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import get_logger

logger = get_logger(__name__)


class DummyModel(nn.Module):
    """Emits random outputs matching GraphGuardModel.forward()'s exact contract.

    fraud_score    : Tensor[n]        in [0, 1]           (torch.rand)
    anomaly_score  : Tensor[n]        >= 0                (torch.rand, unbounded above in the
                                                            real model, so we occasionally emit
                                                            values > 1 to stop consumers from
                                                            silently assuming a [0,1] range)
    embeddings     : Tensor[n, hidden_dim]
    attention      : Tensor[n_edges]  or None             (None on ~20% of calls, since the
                                                            real GAT layer can also return None
                                                            when attention isn't requested --
                                                            C's code must handle both cases)
    """

    def __init__(self, hidden_dim: int = 64, always_return_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.always_return_attention = always_return_attention
        logger.info("DummyModel initialised (hidden_dim=%d) -- outputs are random, not real predictions.", hidden_dim)

    def forward(self, data) -> dict:
        n = int(data.num_nodes)
        n_edges = int(data.edge_index.shape[1]) if getattr(data, "edge_index", None) is not None else 0

        fraud_score = torch.rand(n)
        # anomaly_score is reconstruction error in the real model: non-negative, unbounded.
        # Scale by a few random factors so it isn't mistaken for a [0,1] probability.
        anomaly_score = torch.rand(n) * torch.randint(1, 5, (1,)).float()
        embeddings = torch.randn(n, self.hidden_dim)

        if self.always_return_attention or torch.rand(1).item() > 0.2:
            attention = torch.rand(n_edges)
        else:
            attention = None

        return {
            "fraud_score": fraud_score,
            "anomaly_score": anomaly_score,
            "embeddings": embeddings,
            "attention": attention,
        }

    @classmethod
    def load(cls, checkpoint_path: str) -> "DummyModel":
        """No real checkpoint to load -- DummyModel never trains. Accepts the argument
        only so callers written against GraphGuardModel.load() don't need an if-branch."""
        logger.info("DummyModel.load() called with '%s' -- ignored, returning a fresh DummyModel.", checkpoint_path)
        return cls()
