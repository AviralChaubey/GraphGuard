"""Wraps PyTorch Geometric's `GNNExplainer` behind our own model contract (Contract 2:
`model.forward(data) -> dict`), so anything upstream never has to know PyG's explainer API.

Why GNNExplainer and not just showing attention weights
---------------------------------------------------------
Attention weights (GAT's `attention` output) tell you what the model *looked at* while
computing one forward pass -- they're a byproduct of the architecture. GNNExplainer instead
asks a sharper question: "what is the smallest edge/feature subset that, if I mask everything
else out, the model still produces roughly the same prediction?" It's a post-hoc optimisation
against the model's actual output, not an architectural artifact -- so it works even on models
with no attention mechanism at all, and it directly optimises for the property we actually
want (a small, sufficient explanation), which raw attention does not.

Why not SHAP
-------------
SHAP treats the model as a black box over *feature* coalitions and is expensive to run
per-instance on a graph (subgraph structure isn't naturally a "feature" to permute). GNNExplainer
optimises soft masks over both edges and node features jointly, in the same input space the
model actually consumes (the graph), which is what lets us report both a subgraph *and* a
feature importance ranking from one method.

How we know the explanation is faithful
-----------------------------------------
We don't just trust the mask -- `src/explain/metrics.py` recomputes the prediction with only
the evidence subgraph kept (fidelity) and reports how small that subgraph is (sparsity). Those
numbers, not the mask itself, are what go in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig

from src.utils.config import get_logger, load_config

logger = get_logger(__name__)


def get_node_labels(data) -> Optional[torch.Tensor]:
    """Return per-node labels from `data`, or None if it has neither.

    A's real `to_pyg()` output isn't pinned to a specific attribute name in
    docs/CONTRACTS.md, so we accept the idiomatic PyG convention (`data.y`) first and fall
    back to `data.labels` (what our own `src/_stubs/ring.py` uses). Per
    docs/team/PROMPT_C_explain_api_ui.md, entity nodes (card/device/address/email) carry
    label `-1` and must never be treated as a scoreable target.
    """
    labels = getattr(data, "y", None)
    if labels is None:
        labels = getattr(data, "labels", None)
    return labels


@dataclass
class ExplainerOutput:
    """Raw output of one explanation, before it's turned into evidence/reasons."""

    target_idx: int
    edge_mask: np.ndarray        # [n_edges], aligned to data.edge_index columns, in [0, 1]
    node_feature_mask: np.ndarray  # [n_nodes, n_feat] or [n_feat] depending on mask type
    prediction: float             # model's fraud_score at target_idx, for reference


class _ModelAdapter(torch.nn.Module):
    """Adapts Contract 2's `model.forward(data) -> dict` to the plain
    `forward(x, edge_index) -> Tensor` signature PyG's `Explainer` expects.

    We explain `fraud_score` specifically -- that's the head an analyst cares about; the
    anomaly/embedding/attention outputs aren't targets for explanation.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        from torch_geometric.data import Data

        # edge_attr is optional -- our DummyModel doesn't use it, but B's real model may
        # (e.g. a GATConv keyed on `relation`/`weight`). Threading it through here means the
        # real model's `forward(data)` sees `data.edge_attr` exactly like it would outside
        # the explainer, without any explain/-side code needing to know what the real model
        # does with it.
        data_kwargs = {"x": x, "edge_index": edge_index}
        if edge_attr is not None:
            data_kwargs["edge_attr"] = edge_attr
        data = Data(**data_kwargs)
        out = self.model(data)
        return out["fraud_score"]


class GNNExplainerWrapper:
    """Explain a single node's fraud_score prediction.

    Usage:
        wrapper = GNNExplainerWrapper(model)
        result = wrapper.explain(data, target_idx=3)
    """

    def __init__(self, model: torch.nn.Module, config: Optional[dict] = None):
        full_cfg = load_config()
        cfg = config if config is not None else full_cfg.get("explain", {})
        self.epochs = cfg.get("epochs", 200)
        self.lr = cfg.get("lr", 0.01)
        # GNNExplainer's mask initialisation and Adam optimiser are both stochastic. We
        # reseed torch's global RNG right before every `explain()` call (below) so repeated
        # calls -- in tests, or repeated hits to GET /explain/{id} -- return identical masks
        # instead of drifting run to run.
        self.seed = cfg.get("seed", full_cfg.get("seed", 42))

        # Message passing only ever looks `num_hops` steps around a node, so on big graphs we
        # explain on that neighbourhood alone (same answer, far faster). Small graphs are
        # explained whole, exactly as before.
        self.max_full_graph_nodes = cfg.get("max_full_graph_nodes", 5000)
        from torch_geometric.nn import MessagePassing
        # one hop per message-passing layer; if none are found, be generous rather than wrong
        self.num_hops = sum(isinstance(m, MessagePassing) for m in model.modules()) or 4

        self._adapter = _ModelAdapter(model)
        self._explainer = Explainer(
            model=self._adapter,
            algorithm=GNNExplainer(epochs=self.epochs, lr=self.lr),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=ModelConfig(
                mode="regression",   # fraud_score is a continuous [0,1] score, not a class logit
                task_level="node",
                return_type="raw",
            ),
        )

    def explain(self, data, target_idx: int) -> ExplainerOutput:
        """Run GNNExplainer for a single target node.

        Args:
            data: a `torch_geometric.data.Data` with `.x` and `.edge_index`.
            target_idx: index of the node to explain (must be a scoreable node, i.e. not an
                entity node with label -1 -- see docs/team/PROMPT_C_explain_api_ui.md).

        Returns:
            ExplainerOutput with edge_mask / node_feature_mask as numpy arrays, detached from
            the autograd graph, ready for `src/explain/evidence.py`.
        """
        logger.info("explaining target_idx=%s (epochs=%s)", target_idx, self.epochs)

        # Reseed before every call -- see the comment in __init__. Without this, two calls
        # with the exact same model/data/target_idx can return different edge masks because
        # GNNExplainer's internal mask parameters start from `torch.randn(...)`.
        torch.manual_seed(self.seed)

        explain_kwargs = {}
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None:
            explain_kwargs["edge_attr"] = edge_attr

        n_nodes = int(data.x.size(0))
        if n_nodes > self.max_full_graph_nodes:
            return self._explain_neighbourhood(data, target_idx, explain_kwargs)

        explanation = self._explainer(
            x=data.x, edge_index=data.edge_index, index=target_idx, **explain_kwargs
        )

        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        node_feature_mask = explanation.node_mask.detach().cpu().numpy()
        # `explanation.prediction` is the *full* model output (one value per node), not just
        # the target's -- PyG's `index=` only tells the algorithm which node to explain
        # against, it doesn't narrow what gets stored on the returned Explanation. Index it
        # ourselves.
        raw_prediction = explanation.prediction
        if raw_prediction.numel() > 1:
            raw_prediction = raw_prediction[target_idx]
        prediction = float(raw_prediction.detach().cpu().item())

        return ExplainerOutput(
            target_idx=target_idx,
            edge_mask=edge_mask,
            node_feature_mask=node_feature_mask,
            prediction=prediction,
        )

    def _explain_neighbourhood(self, data, target_idx: int, explain_kwargs: dict) -> ExplainerOutput:
        """Explain on the `num_hops`-hop neighbourhood of the target, then map the masks back
        to the full graph's indexing (edges/nodes outside the neighbourhood get 0)."""
        from torch_geometric.utils import k_hop_subgraph

        subset, sub_edge_index, mapping, sub_edge_bool = k_hop_subgraph(
            int(target_idx), self.num_hops, data.edge_index,
            relabel_nodes=True, num_nodes=int(data.x.size(0)),
        )
        local_idx = int(mapping[0])
        kwargs = {}
        if "edge_attr" in explain_kwargs:
            kwargs["edge_attr"] = explain_kwargs["edge_attr"][sub_edge_bool]

        explanation = self._explainer(
            x=data.x[subset], edge_index=sub_edge_index, index=local_idx, **kwargs
        )

        n_edges = int(data.edge_index.size(1))
        edge_mask = np.zeros(n_edges, dtype=np.float32)
        edge_mask[sub_edge_bool.cpu().numpy()] = explanation.edge_mask.detach().cpu().numpy()

        sub_node_mask = explanation.node_mask.detach().cpu().numpy()
        node_feature_mask = np.zeros((int(data.x.size(0)), sub_node_mask.shape[-1]), dtype=np.float32)
        node_feature_mask[subset.cpu().numpy()] = sub_node_mask

        raw_prediction = explanation.prediction
        if raw_prediction.numel() > 1:
            raw_prediction = raw_prediction[local_idx]
        return ExplainerOutput(
            target_idx=int(target_idx),
            edge_mask=edge_mask,
            node_feature_mask=node_feature_mask,
            prediction=float(raw_prediction.detach().cpu().item()),
        )

    def explain_ring(self, data, ring) -> ExplainerOutput:
        """Explain a ring by explaining its highest-scoring *scoreable* member node.

        Rings don't have a single forward-pass output to explain, so we pick the node whose
        fraud_score is highest under the current model and explain that -- it's the node an
        analyst would click on first, and its evidence subgraph typically pulls in the shared
        entity (device/address) that ties the ring together anyway.

        Entity nodes (card/device/address/email, label `-1`) are excluded from this
        selection: per docs/team/PROMPT_C_explain_api_ui.md they're conduits, not things to
        classify, and were never trained against a fraud label, so their fraud_score is
        meaningless to rank against and must never be picked as the explanation target.
        """
        with torch.no_grad():
            scores = self._adapter.model(data)["fraud_score"]

        labels = get_node_labels(data)
        if labels is not None:
            scoreable_ids: List[int] = [n for n in ring.node_ids if int(labels[n]) != -1]
            if not scoreable_ids:
                raise ValueError(
                    f"ring {getattr(ring, 'ring_id', '?')!r} has no scoreable members -- "
                    "every node in it has label -1 (entity-only ring)"
                )
        else:
            logger.warning(
                "data has neither .y nor .labels -- cannot exclude entity nodes for ring %s; "
                "considering all ring members. This should not happen once A's real "
                "GraphBundle/to_pyg() output is wired in.",
                getattr(ring, "ring_id", "?"),
            )
            scoreable_ids = list(ring.node_ids)

        ring_scores = scores[scoreable_ids]
        best_local_idx = int(torch.argmax(ring_scores))
        target_idx = scoreable_ids[best_local_idx]
        return self.explain(data, target_idx)
