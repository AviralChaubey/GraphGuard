"""Train GraphGuardModel: weighted BCE for the fraud head (class imbalance), MSE
reconstruction loss for the autoencoder head trained on licit-only nodes, early stopping,
checkpointing to results/checkpoints/.

A's src/data and src/graph aren't merged yet, so this script falls back to a synthetic
graph when the real import fails -- loud warning, not a silent substitution, so nobody
mistakes a demo run for a real result. Swap in the real pipeline the moment A lands; the
model/training code itself does not change.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --synthetic   # force demo mode
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

from src.models.graphguard import GraphGuardModel
from src.utils.config import get_logger, load_config, set_seed

logger = get_logger(__name__)


def make_synthetic_dataset(n_nodes: int = 500, n_edges: int = 2000, n_feat: int = 16, fraud_rate: float = 0.05, unknown_rate: float = 0.3):
    """Stand-in for `to_pyg(load_dataset(...))` until A's pipeline lands. Produces labels
    with the same -1/0/1 convention as the real contract, and a temporal-ish split (first
    70% of node index = train, last 30% = test) so the training loop exercises the real
    masking logic, not a shortcut.
    """
    x = torch.randn(n_nodes, n_feat)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))

    labels = torch.zeros(n_nodes, dtype=torch.long)
    n_fraud = int(n_nodes * fraud_rate)
    n_unknown = int(n_nodes * unknown_rate)
    perm = torch.randperm(n_nodes)
    labels[perm[:n_fraud]] = 1
    labels[perm[n_fraud:n_fraud + n_unknown]] = -1
    # everyone else stays 0 (licit)

    split_point = int(n_nodes * 0.7)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:split_point] = True
    test_mask[split_point:] = True
    # -1 (unknown) must never be trainable or testable as if it were licit
    train_mask &= labels != -1
    test_mask &= labels != -1

    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=n_nodes)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


def load_real_or_synthetic(dataset_name: str, cfg: dict):
    try:
        from src.graph import load_pyg

        data = load_pyg(dataset_name)
        logger.info("Loaded real dataset '%s' via A's pipeline.", dataset_name)
        return data
    except ImportError:
        logger.warning(
            "src.data / src.graph not available yet (A's pipeline isn't merged) -- "
            "falling back to a SYNTHETIC graph. Any metrics from this run are a pipeline "
            "smoke test, not a real result. Do not put these numbers in the report."
        )
        return make_synthetic_dataset()


def compute_pos_weight(labels: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """pos_weight for BCEWithLogitsLoss = n_negative / n_positive among *labeled* training
    nodes only (-1 excluded). Recomputed from actual data rather than trusting the config
    placeholder, since the placeholder is dataset-specific and easy to forget to update."""
    train_labels = labels[train_mask]
    n_pos = (train_labels == 1).sum().item()
    n_neg = (train_labels == 0).sum().item()
    if n_pos == 0:
        logger.warning("No positive (fraud) labels in the training mask -- using pos_weight=1.0")
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos)


def _is_hetero(data) -> bool:
    return hasattr(data, "x_dict")


def _target(data):
    """For a HeteroData, the target node type's own Storage object (where y/x/masks
    live); for a homogeneous Data, data itself. Lets the rest of train() read
    y/x/train_mask/test_mask the same way regardless of graph shape."""
    if _is_hetero(data):
        return data[getattr(data, "target_node_type", "transaction")]
    return data


def train(cfg: dict, data, checkpoint_dir: Path) -> GraphGuardModel:
    model_cfg = cfg["model"]
    ae_cfg = cfg["autoencoder"]
    hetero = _is_hetero(data)
    tgt = _target(data)

    if hetero:
        target_nt = getattr(data, "target_node_type", "transaction")
        in_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
        model = GraphGuardModel(
            hidden_dim=model_cfg["hidden_dim"],
            num_sage_layers=model_cfg["num_sage_layers"],
            num_gat_layers=model_cfg["num_gat_layers"],
            gat_heads=model_cfg["gat_heads"],
            dropout=model_cfg["dropout"],
            latent_dim=ae_cfg["latent_dim"],
            node_types=list(data.node_types),
            edge_types=list(data.edge_types),
            target_node_type=target_nt,
            in_dims=in_dims,
        )
    else:
        model = GraphGuardModel(
            in_dim=data.x.shape[1],
            hidden_dim=model_cfg["hidden_dim"],
            num_sage_layers=model_cfg["num_sage_layers"],
            num_gat_layers=model_cfg["num_gat_layers"],
            gat_heads=model_cfg["gat_heads"],
            dropout=model_cfg["dropout"],
            latent_dim=ae_cfg["latent_dim"],
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_cfg["learning_rate"], weight_decay=model_cfg["weight_decay"]
    )

    pos_weight = compute_pos_weight(tgt.y, tgt.train_mask)
    logger.info("Computed pos_weight=%.2f from training labels.", pos_weight.item())

    # Autoencoder trains only on licit (label==0) nodes within the train split -- fraud
    # nodes should reconstruct *badly*, that's the anomaly signal, so they must never be
    # part of the reconstruction loss.
    licit_train_mask = tgt.train_mask & (tgt.y == 0)

    # Checkpoint selection and early stopping track val PR-AUC, not val BCE loss.
    # Found the hard way: on Elliptic, val BCE bottoms out around epoch 4-5 and then
    # climbs (classic overfitting-on-the-loss), but val PR-AUC keeps improving for
    # 60+ more epochs after that. Selecting by BCE alone was silently saving a
    # checkpoint from epoch 4 when the real best epoch (by the metric we actually
    # report, per docs/EVALUATION.md) was ~70 epochs later -- PR-AUC 0.22 vs 0.66
    # on the same run. BCE and PR-AUC are not interchangeable as stopping criteria
    # under heavy class imbalance + pos_weight; always track the real target metric.
    best_val_pr_auc = float("-inf")
    patience_counter = 0
    patience = model_cfg["early_stopping_patience"]

    for epoch in range(model_cfg["epochs"]):
        model.train()
        optimizer.zero_grad()

        out = model.forward_with_logits(data)

        fraud_mask = tgt.train_mask  # already excludes -1
        fraud_loss = F.binary_cross_entropy_with_logits(
            out["fraud_logits"][fraud_mask],
            tgt.y[fraud_mask].float(),
            pos_weight=pos_weight,
        )

        if licit_train_mask.sum() > 0:
            recon_loss = F.mse_loss(out["reconstruction"][licit_train_mask], tgt.x[licit_train_mask])
        else:
            recon_loss = torch.tensor(0.0)

        loss = fraud_loss + recon_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model.forward_with_logits(data)
            val_mask = tgt.test_mask
            val_loss = F.binary_cross_entropy_with_logits(
                val_out["fraud_logits"][val_mask], tgt.y[val_mask].float(), pos_weight=pos_weight
            ).item()
            val_scores = val_out["fraud_score"][val_mask].cpu().numpy()
            val_labels = tgt.y[val_mask].cpu().numpy()
            val_pr_auc = average_precision_score(val_labels, val_scores)

        logger.info(
            "epoch %03d | train_loss=%.4f (fraud=%.4f recon=%.4f) | val_loss=%.4f | val_pr_auc=%.4f",
            epoch, loss.item(), fraud_loss.item(), recon_loss.item(), val_loss, val_pr_auc,
        )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if hetero:
                model_kwargs = {
                    "hidden_dim": model_cfg["hidden_dim"],
                    "num_sage_layers": model_cfg["num_sage_layers"],
                    "num_gat_layers": model_cfg["num_gat_layers"],
                    "gat_heads": model_cfg["gat_heads"],
                    "dropout": model_cfg["dropout"],
                    "latent_dim": ae_cfg["latent_dim"],
                    "node_types": list(data.node_types),
                    "edge_types": list(data.edge_types),
                    "target_node_type": target_nt,
                    "in_dims": in_dims,
                }
            else:
                model_kwargs = {
                    "in_dim": data.x.shape[1],
                    "hidden_dim": model_cfg["hidden_dim"],
                    "num_sage_layers": model_cfg["num_sage_layers"],
                    "num_gat_layers": model_cfg["num_gat_layers"],
                    "gat_heads": model_cfg["gat_heads"],
                    "dropout": model_cfg["dropout"],
                    "latent_dim": ae_cfg["latent_dim"],
                }
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_pr_auc": val_pr_auc,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (best val_pr_auc=%.4f)", epoch, best_val_pr_auc)
                break

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data even if A's pipeline is available")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    if args.synthetic:
        data = make_synthetic_dataset()
        logger.info("Forced synthetic dataset via --synthetic flag.")
   
    else:
        data = load_real_or_synthetic(cfg["data"]["dataset"], cfg)   
    # Per-dataset subfolder: without this, training a second dataset against the
    # same config silently overwrites the first dataset's checkpoint at the same
    # path (results/checkpoints/best.pt) with no warning. Found this the hard way
    # running Elliptic then PaySim back to back.
    dataset_name = "synthetic" if args.synthetic else cfg["data"]["dataset"]
    checkpoint_dir = Path(cfg["checkpoints"]["dir"]) / dataset_name
    train(cfg, data, checkpoint_dir)
             

if __name__ == "__main__":
    main()


    
