"""GraphSAGE -> GAT -> dual heads (fraud classifier + autoencoder).

Contract (docs/CONTRACTS.md, Contract 2): forward() returns a dict with the exact same
keys/shapes/ranges as DummyModel, so C's code never has to change when this replaces it.

Homogeneous graphs only for now (Elliptic). IEEE-CIS needs HeteroConv -- see the TODO at the
bottom; don't build that until the homogeneous path is validated on Elliptic.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv

from src.utils.config import get_logger

logger = get_logger(__name__)


class GraphGuardModel(nn.Module):
    """GraphSAGE encoder (inductive) -> GAT attention layers -> two heads.

    Head A (fraud classifier): linear -> sigmoid over the final embeddings.
    Head B (autoencoder): decodes embeddings back to input feature space; reconstruction
    error is the anomaly score. Trained on licit-only nodes at fit time (see scripts/train.py),
    but forward() always scores every node -- the loss mask decides what's used to update
    weights, not what's scored at inference.

    Why GraphSAGE then GAT, not just one or the other:
    - GraphSAGE alone is inductive (generalises to brand-new accounts with no retraining --
      Van Belle et al. 2022) but averages/pools neighbours uniformly.
    - GAT alone attends over neighbours but its standard formulation is transductive-leaning
      and less proven at generalising to unseen nodes cold.
    - Stacking them gets inductive generalisation from SAGE first, then lets GAT re-weight
      which neighbours matter most -- important because fraud rings camouflage by attaching to
      many legitimate accounts (heterophily; Ren et al. 2023), so uniform averaging dilutes the
      fraud signal. Sheu & Li 2022 validate GAT specifically on this problem/datasets.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_sage_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.3,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # --- GraphSAGE encoder ---
        self.sage_layers = nn.ModuleList()
        sage_in = in_dim
        for _ in range(num_sage_layers):
            self.sage_layers.append(SAGEConv(sage_in, hidden_dim))
            sage_in = hidden_dim

        # --- GAT attention layers ---
        # GATv2Conv (Brody et al. 2022) fixes the static-attention limitation of the original
        # GAT -- strictly better, same interface, worth the upgrade over vanilla GATConv.
        self.gat_layers = nn.ModuleList()
        gat_in = hidden_dim
        for i in range(num_gat_layers):
            is_last = i == num_gat_layers - 1
            heads = 1 if is_last else gat_heads
            concat = not is_last
            self.gat_layers.append(
                GATv2Conv(gat_in, hidden_dim, heads=heads, concat=concat, dropout=dropout)
            )
            gat_in = hidden_dim * heads if concat else hidden_dim

        # --- Head A: fraud classifier ---
        self.fraud_head = nn.Linear(hidden_dim, 1)

        # --- Head B: autoencoder (novel-anomaly branch) ---
        self.encoder_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

        logger.info(
            "GraphGuardModel built: in_dim=%d hidden_dim=%d sage_layers=%d gat_layers=%d heads=%d",
            in_dim, hidden_dim, num_sage_layers, num_gat_layers, gat_heads,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run SAGE then GAT, returning (embeddings, last-layer attention weights)."""
        h = x
        for layer in self.sage_layers:
            h = F.relu(layer(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)

        attention = None
        for i, layer in enumerate(self.gat_layers):
            is_last = i == len(self.gat_layers) - 1
            if is_last:
                h, (_, attn_weights) = layer(h, edge_index, return_attention_weights=True)
                # attn_weights: [n_edges, heads] with heads=1 on the last layer -> squeeze to [n_edges]
                attention = attn_weights.squeeze(-1)
            else:
                h = layer(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h, attention

    def forward(self, data) -> dict:
        x, edge_index = data.x, data.edge_index
        embeddings, attention = self.encode(x, edge_index)

        fraud_logits = self.fraud_head(embeddings).squeeze(-1)
        fraud_score = torch.sigmoid(fraud_logits)

        latent = self.encoder_to_latent(embeddings)
        reconstruction = self.decoder(latent)
        # per-node reconstruction error (mean squared error across feature dim), >= 0
        anomaly_score = F.mse_loss(reconstruction, x, reduction="none").mean(dim=-1)

        return {
            "fraud_score": fraud_score,
            "anomaly_score": anomaly_score,
            "embeddings": embeddings,
            "attention": attention,
        }

    def forward_with_logits(self, data) -> dict:
        """Same as forward(), but also returns raw fraud logits (pre-sigmoid) and the
        reconstruction, since the training loop needs logits for BCEWithLogitsLoss and the
        raw reconstruction for the autoencoder loss -- forward() alone only exposes what
        Contract 2 requires downstream."""
        x, edge_index = data.x, data.edge_index
        embeddings, attention = self.encode(x, edge_index)

        fraud_logits = self.fraud_head(embeddings).squeeze(-1)
        latent = self.encoder_to_latent(embeddings)
        reconstruction = self.decoder(latent)
        anomaly_score = F.mse_loss(reconstruction, x, reduction="none").mean(dim=-1)

        return {
            "fraud_score": torch.sigmoid(fraud_logits),
            "fraud_logits": fraud_logits,
            "anomaly_score": anomaly_score,
            "reconstruction": reconstruction,
            "embeddings": embeddings,
            "attention": attention,
        }

    @classmethod
    def load(cls, checkpoint_path: str) -> "GraphGuardModel":
        """Load a checkpoint saved by scripts/train.py. The checkpoint stores both the
        state_dict and the constructor kwargs, so the architecture doesn't need to be
        guessed or hardcoded at load time."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        logger.info("Loaded GraphGuardModel from %s", path)
        return model

# TODO(B): HeteroConv variant for IEEE-CIS (card/device/ip node types) once the homogeneous
# path is validated on Elliptic. Swap SAGEConv/GATv2Conv for their Hetero-wrapped equivalents
# keyed by edge type; the forward()/Contract-2 output shape does not change.
