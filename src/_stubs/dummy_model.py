"""Placeholder for B's `src/models/dummy.py`.

Per docs/CONTRACTS.md, B commits the real `src/models/dummy.py` on day one. This file exists
only so my code (and tests) can run *today* without waiting -- delete this once B's real file
lands in `src/models/dummy.py` and repoint the imports in `src/explain/` and `src/api/` there.

Implements exactly the frozen Contract 2 signature:

    model.forward(data) -> {
        'fraud_score':   Tensor[n] in [0,1],
        'anomaly_score': Tensor[n] >= 0,
        'embeddings':    Tensor[n, hidden_dim],
        'attention':     Tensor[n_edges] or None,
    }

Why this isn't just `torch.rand(...)`
---------------------------------------
An earlier version of this stub returned pure random tensors, independent of `data.x` and
`data.edge_index`, redrawn from a stateful `Generator` on every call. That's wrong in two
ways that matter beyond "looks lazy":

1. **It broke GNNExplainer.** GNNExplainer works by optimising an edge/feature mask against
   the model's *gradient* -- it needs `d(fraud_score)/d(edge_mask)` to be a real, structured
   signal. A model whose output doesn't depend on its input at all has zero gradient w.r.t.
   everything, so the "explanation" GNNExplainer produces is just whatever its random mask
   initialisation happened to be. Every downstream fidelity/sparsity number was meaningless.
2. **It broke reproducibility.** Because the old stub redrew from a persistent `Generator`
   each call, `model(data)` gave a *different* answer every time even for the identical
   input -- two calls to `/score/{id}` a second apart returned different scores.

This version is a genuine (if untrained) 2-layer GraphSAGE encoder: `SAGEConv` is a real
`torch_geometric.nn.MessagePassing` layer, which matters specifically because PyG's
`Explainer` intercepts edge importance by hooking `MessagePassing.propagate` -- a
hand-rolled `scatter_mean` that doesn't subclass `MessagePassing` would silently ignore the
explainer's edge mask entirely. Weights are fixed once at construction from `seed`, so the
model is a pure, deterministic function of `(x, edge_index)` afterwards: same input, same
output, every time -- not trained, but genuinely graph-aware.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class DummyModel(nn.Module):
    def __init__(self, in_dim: Optional[int] = None, hidden_dim: int = 16, seed: int = 42):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.in_dim: Optional[int] = None
        self.conv1: Optional[SAGEConv] = None
        self.conv2: Optional[SAGEConv] = None
        self.fraud_head: Optional[nn.Linear] = None
        self.anomaly_decoder: Optional[nn.Linear] = None
        if in_dim is not None:
            self._build(in_dim)

    def _build(self, in_dim: int) -> None:
        """Construct layers with fixed, seeded weights. Called lazily on first `forward()`
        if `in_dim` wasn't known at construction time, so `DummyModel()` (no args) still
        works against whatever feature width the caller's graph happens to have.
        """
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(self.seed)
            self.conv1 = SAGEConv(in_dim, self.hidden_dim)
            self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
            self.fraud_head = nn.Linear(self.hidden_dim, 1)
            self.anomaly_decoder = nn.Linear(self.hidden_dim, in_dim)
        finally:
            # don't let building this model perturb whatever seed the caller (e.g.
            # GNNExplainerWrapper) had set up around it.
            torch.random.set_rng_state(rng_state)
        self.in_dim = in_dim

    def forward(self, data) -> dict:
        x, edge_index = data.x, data.edge_index
        if self.conv1 is None:
            self._build(x.size(1))
        elif x.size(1) != self.in_dim:
            raise ValueError(
                f"DummyModel was built for in_dim={self.in_dim}, got x with {x.size(1)} features"
            )

        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)  # -> embeddings

        fraud_score = torch.sigmoid(self.fraud_head(h)).squeeze(-1)

        # Reconstruction-error anomaly head: how well does the aggregated embedding predict
        # this node's own original features? Bigger residual = more anomalous. Deterministic
        # and differentiable, like everything else here.
        recon = self.anomaly_decoder(h)
        anomaly_score = (recon - x).pow(2).mean(dim=1)

        src, dst = edge_index
        attention = torch.sigmoid((h[src] * h[dst]).sum(dim=1))

        return {
            "fraud_score": fraud_score,
            "anomaly_score": anomaly_score,
            "embeddings": h,
            "attention": attention,
        }

    @classmethod
    def load(cls, checkpoint_path: str) -> "DummyModel":
        return cls()
