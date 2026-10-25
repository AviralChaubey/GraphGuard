"""GraphSAGE -> GAT -> dual heads (fraud classifier + autoencoder).

Contract (docs/CONTRACTS.md, Contract 2): forward() returns a dict with the exact same
keys/shapes/ranges as DummyModel, so C's code never has to change when this replaces it.

Two graph shapes are supported:
  - Homogeneous (Elliptic, PaySim): data.x / data.edge_index, as before.
  - Heterogeneous (IEEE-CIS: transaction/card/device/address/email): data.x_dict /
    data.edge_index_dict (a torch_geometric HeteroData). This was the TODO at the bottom
    of this file -- see below for what changed and, importantly, what this does NOT cover.

HeteroConv notes:
  - Each layer is a HeteroConv wrapping one SAGEConv (or GATv2Conv) per edge type, with
    lazy (-1, -1) input dims -- node types enter the first layer with different feature
    widths (transaction=34, card/device/address/email=5), so a single shared weight matrix
    isn't possible at layer 0; lazy init resolves each edge type's own weights on first call.
  - to_pyg() already adds the reverse of every relation, so every node type is the
    destination of at least one edge type each layer and gets a fresh embedding -- without
    that, entity node types would never update past their raw input features.
  - Attention-weight extraction (Contract 2's `attention` key) is skipped in the hetero
    path and always returns None, which the contract allows ("Tensor[n_edges] or None").
    Per-edge-type attention across a HeteroConv's multiple relations doesn't have one
    single "last layer's edge weights" the way the homogeneous path does; extracting it
    meaningfully is a bigger, separate piece of work, not bundled into this fix.
  - add_self_loops=False on the hetero GAT layers: PyG's GATv2Conv self-loop logic assumes
    source and destination are the same node set, which isn't true for most of our edge
    types (transaction -> card, etc.) and errors otherwise.

KNOWN GAP THIS DOES NOT FIX: src/ring/extract.py's extract_rings() still assumes a single
homogeneous data.edge_index / data.num_nodes, so ring extraction on IEEE-CIS needs its own
follow-up (building the induced subgraph over the target node type's *derived* adjacency --
e.g. two transactions sharing a card -- rather than the raw heterogeneous edge list). This
fix unblocks training and detection metrics (PR-AUC/F1) on IEEE-CIS; ring quality on IEEE-CIS
is still open.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

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
        in_dim: int = 0,
        hidden_dim: int = 64,
        num_sage_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.3,
        latent_dim: int = 32,
        node_types: list[str] | None = None,
        edge_types: list[tuple] | None = None,
        target_node_type: str = "transaction",
        in_dims: dict | None = None,
    ):
        """in_dim: feature width for the homogeneous path (ignored when node_types is set).
        node_types/edge_types: pass both to build the heterogeneous (HeteroConv) variant --
        node_types is every node type's name, edge_types is every (src, relation, dst) triple
        INCLUDING the reverse relations to_pyg() adds. in_dims maps each node type to its own
        feature width (they differ -- e.g. IEEE-CIS transaction=34, card/device/address/email=5).
        Built from explicit per-type dims rather than PyG's lazy (-1, -1) in_channels so a
        saved checkpoint's shapes are known up front and load() doesn't need a dummy forward
        pass just to materialize lazy modules before load_state_dict().
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.hetero = node_types is not None and edge_types is not None
        self.target_node_type = target_node_type

        if self.hetero:
            self.node_types = list(node_types)
            self.edge_types = [tuple(et) for et in edge_types]
            if in_dims is None:
                raise ValueError("in_dims is required when building the hetero variant")
            recon_in_dim = in_dims[target_node_type]

            self.sage_layers = nn.ModuleList()
            for layer_idx in range(num_sage_layers):
                convs = {}
                for et in self.edge_types:
                    src_t, _, dst_t = et
                    in_ch = (in_dims[src_t], in_dims[dst_t]) if layer_idx == 0 else hidden_dim
                    convs[et] = SAGEConv(in_ch, hidden_dim)
                self.sage_layers.append(HeteroConv(convs, aggr="sum"))

            self.gat_layers = nn.ModuleList()
            # Width feeding the first GAT layer: hidden_dim if any SAGE layer ran (SAGEConv's
            # out_channels is always hidden_dim, heads don't apply to it), otherwise each
            # type's own raw width -- same two cases the homogeneous path handles via
            # `gat_in = hidden_dim * heads if concat else hidden_dim` carried across layers;
            # mirrored here per edge type since GAT's own concat=True also changes the width
            # a *subsequent* GAT layer must accept.
            gat_in_uniform = hidden_dim if num_sage_layers > 0 else None
            for i in range(num_gat_layers):
                is_last = i == num_gat_layers - 1
                heads = 1 if is_last else gat_heads
                concat = not is_last
                convs = {}
                for et in self.edge_types:
                    src_t, _, dst_t = et
                    if gat_in_uniform is None:
                        in_ch = (in_dims[src_t], in_dims[dst_t])
                    else:
                        in_ch = gat_in_uniform
                    convs[et] = GATv2Conv(in_ch, hidden_dim, heads=heads, concat=concat,
                                           dropout=dropout, add_self_loops=False)
                self.gat_layers.append(HeteroConv(convs, aggr="sum"))
                gat_in_uniform = hidden_dim * heads if concat else hidden_dim
        else:
            recon_in_dim = in_dim

            # --- GraphSAGE encoder ---
            self.sage_layers = nn.ModuleList()
            sage_in = in_dim
            for _ in range(num_sage_layers):
                self.sage_layers.append(SAGEConv(sage_in, hidden_dim))
                sage_in = hidden_dim

            # --- GAT attention layers ---
            # GATv2Conv (Brody et al. 2022) fixes the static-attention limitation of the
            # original GAT -- strictly better, same interface, worth the upgrade over vanilla
            # GATConv.
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
            nn.Linear(hidden_dim, recon_in_dim),
        )

        logger.info(
            "GraphGuardModel built: hetero=%s hidden_dim=%d sage_layers=%d gat_layers=%d heads=%d",
            self.hetero, hidden_dim, num_sage_layers, num_gat_layers, gat_heads,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Homogeneous path: run SAGE then GAT, returning (embeddings, last-layer attention)."""
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

    def encode_hetero(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Heterogeneous path: run each layer's HeteroConv across every edge type, keeping
        the prior embedding for any node type a layer doesn't happen to touch (defensive --
        with to_pyg()'s forward+reverse edges every type is a destination somewhere, but a
        graph missing some relation shouldn't hard-crash the whole model)."""
        h_dict = x_dict
        for layer in self.sage_layers:
            new_h = layer(h_dict, edge_index_dict)
            new_h = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                     for k, v in new_h.items()}
            h_dict = {**h_dict, **new_h}
        for layer in self.gat_layers:
            new_h = layer(h_dict, edge_index_dict)
            new_h = {k: F.dropout(F.elu(v), p=self.dropout, training=self.training)
                     for k, v in new_h.items()}
            h_dict = {**h_dict, **new_h}
        return h_dict

    def _encode_any(self, data):
        """Returns (embeddings_for_target_type, attention, x_for_target_type)."""
        if self.hetero:
            target_nt = getattr(data, "target_node_type", self.target_node_type)
            h_dict = self.encode_hetero(data.x_dict, data.edge_index_dict)
            return h_dict[target_nt], None, data.x_dict[target_nt]
        embeddings, attention = self.encode(data.x, data.edge_index)
        return embeddings, attention, data.x

    def forward(self, data) -> dict:
        embeddings, attention, x = self._encode_any(data)

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
        embeddings, attention, x = self._encode_any(data)

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
