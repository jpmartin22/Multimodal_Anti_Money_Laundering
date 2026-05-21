"""
train_graphsage.py
==================
Member A (Jaya) — Week 2 GraphSAGE training on Elliptic transaction graph.

Trains a 2-layer GraphSAGE encoder on the preprocessed graph from
elliptic_preprocess.py and logs everything to MLflow.

Architecture (matches proposal exactly)
----------------------------------------
    Input  : (N, 165)  node feature matrix
    SAGE 1 : 165 → 128, ReLU, Dropout 0.3
    SAGE 2 : 128 → 64
    Output : 64-dimensional embedding  ← used by fusion head
    Head   : 64 → 1, sigmoid           ← used for standalone eval

Training strategy: full-graph (all 46k nodes fit in memory).
    - Masks split nodes into train / val / test sets (stratified).
    - Loss computed only on labeled train nodes.
    - Class imbalance handled via BCEWithLogitsLoss(pos_weight).

Phase 2 additions
-----------------
    - Python logger (DEBUG/INFO/WARNING/ERROR) with log rotation
    - NaN detection + shape assertion checks
    - Timing logs per epoch and total training
    - Error logging with full traceback

Usage
-----
    # Full training (~5 min CPU):
    python train_graphsage.py

    # Smoke test (fast, for CI):
    python train_graphsage.py --epochs 2 --max_nodes 8000

    # Custom hyperparams (for MLflow experiments):
    python train_graphsage.py --lr 0.0005 --hidden_dim 256 --epochs 100
"""

import argparse
import json
import logging
import logging.handlers
import os
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# ─────────────────────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER SETUP  (Phase 2 — §2 Monitoring & §5 Logging)
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(log_file: str = "logs/graphsage_training.log") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("graphsage")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt_console = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt_console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt_file)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────


class GraphSAGEEncoder(nn.Module):
    """
    2-layer GraphSAGE encoder.

    Input  : (N, 165) node feature matrix + edge_index
    Output : (N, 64)  node embeddings  ← used by fusion head
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
        self.dropout = dropout
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGEClassifier(nn.Module):
    """
    Full classifier: GraphSAGEEncoder + binary classification head.
    The encoder is extracted separately for the fusion head.
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(embedding_dim, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x, edge_index)
        logit = self.head(embedding).squeeze(-1)
        return logit, embedding


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECKS  (Phase 2 — §2 Model Assertion Checks)
# ─────────────────────────────────────────────────────────────────────────────


def run_sanity_checks(
    features: np.ndarray, labels: np.ndarray, edge_index: np.ndarray
):
    logger.info("Running sanity checks on graph data...")

    assert features.ndim == 2, f"Expected 2D features (N, F), got {features.ndim}D"
    assert labels.ndim == 1, f"Expected 1D labels, got {labels.ndim}D"
    assert features.shape[0] == labels.shape[0], (
        f"Node count mismatch: {features.shape[0]} features vs {labels.shape[0]} labels"
    )
    assert edge_index.shape[0] == 2, (
        f"edge_index must have shape (2, E), got {edge_index.shape}"
    )
    logger.debug(
        f"Shape checks passed — features: {features.shape}, "
        f"labels: {labels.shape}, edges: {edge_index.shape[1]:,}"
    )

    nan_count = np.isnan(features).sum()
    assert nan_count == 0, f"NaN detected in features — {nan_count} NaN values"
    logger.debug("NaN check passed")

    unique_labels = np.unique(labels)
    assert set(unique_labels).issubset({0, 1}), (
        f"Labels must be 0 or 1, found: {unique_labels}"
    )

    fraud_rate = labels.mean()
    logger.info(
        f"Sanity checks passed — {len(features):,} nodes | "
        f"{edge_index.shape[1]:,} edges | "
        f"fraud rate: {fraud_rate*100:.2f}% ({labels.sum():,} illicit)"
    )
    if fraud_rate < 0.05:
        logger.warning(
            f"Low fraud rate ({fraud_rate*100:.2f}%) — pos_weight will handle imbalance"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def load_graph(
    features_path: str,
    edge_index_path: str,
    labels_path: str,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info(f"Loading graph data from {os.path.dirname(features_path)}/")
    t0 = time.time()

    for path in [features_path, edge_index_path, labels_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"Missing: {path}")

    features = np.load(features_path).astype(np.float32)
    edge_index = np.load(edge_index_path)
    labels = np.load(labels_path)

    logger.debug(f"Raw load time: {time.time()-t0:.2f}s")
    logger.info(
        f"Loaded — nodes: {features.shape[0]:,} | "
        f"features: {features.shape[1]} | edges: {edge_index.shape[1]:,}"
    )

    if max_nodes and max_nodes < len(features):
        # Stratified subsample for smoke tests
        idx_fraud = np.where(labels == 1)[0]
        idx_legit = np.where(labels == 0)[0]
        n_fraud = min(len(idx_fraud), max(1, int(max_nodes * labels.mean())))
        n_legit = max_nodes - n_fraud
        keep = np.concatenate([
            np.random.choice(idx_fraud, n_fraud, replace=False),
            np.random.choice(idx_legit, n_legit, replace=False),
        ])
        keep_set = set(keep.tolist())

        # Remap node indices for edge_index
        old_to_new = {old: new for new, old in enumerate(keep)}
        mask_edges = np.array([
            (s in keep_set and d in keep_set)
            for s, d in zip(edge_index[0], edge_index[1])
        ])
        ei_sub = edge_index[:, mask_edges]
        ei_sub = np.array([[old_to_new[v] for v in ei_sub[0]],
                           [old_to_new[v] for v in ei_sub[1]]])

        features = features[keep]
        labels = labels[keep]
        edge_index = ei_sub
        logger.info(
            f"Subsampled to {max_nodes:,} nodes | "
            f"remaining edges: {edge_index.shape[1]:,}"
        )

    return features, edge_index, labels


def build_masks(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 split → boolean node masks."""
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.15, stratify=labels, random_state=SEED
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=0.15 / 0.85,
        stratify=labels[idx_trainval],
        random_state=SEED,
    )

    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    logger.info(
        f"Split — Train: {train_mask.sum():,} | "
        f"Val: {val_mask.sum():,} | Test: {test_mask.sum():,}"
    )
    return train_mask, val_mask, test_mask


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    model: GraphSAGEClassifier,
    data: Data,
    mask: torch.Tensor,
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        labels_np = data.y[mask].cpu().numpy()

    if np.isnan(probs).any():
        logger.error("NaN detected in model predictions")
        raise ValueError("NaN in predictions")

    logger.debug(
        f"Prob stats — min:{probs.min():.4f} max:{probs.max():.4f} "
        f"mean:{probs.mean():.4f} median:{np.median(probs):.4f}"
    )

    auc_pr = average_precision_score(labels_np, probs)

    precision, recall, thresholds = precision_recall_curve(labels_np, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])

    preds = (probs >= best_threshold).astype(int)

    idx_r80 = np.searchsorted(recall[::-1], 0.8)
    prec_at_r80 = float(precision[::-1][idx_r80]) if idx_r80 < len(precision) else 0.0

    report = classification_report(labels_np, preds, output_dict=True, zero_division=0)

    return {
        "auc_pr": round(float(auc_pr), 4),
        "prec_at_recall_80": round(prec_at_r80, 4),
        "f1_fraud": round(report.get("1", {}).get("f1-score", 0), 4),
        "recall_fraud": round(report.get("1", {}).get("recall", 0), 4),
        "precision_fraud": round(report.get("1", {}).get("precision", 0), 4),
        "false_positive_rate": round(
            1 - report.get("0", {}).get("recall", 1.0), 4
        ),
        "accuracy": round(report.get("accuracy", 0), 4),
        "optimal_threshold": round(best_threshold, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────


def train(args):
    train_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    features, edge_index, labels = load_graph(
        args.features, args.edge_index, args.labels, args.max_nodes
    )
    run_sanity_checks(features, labels, edge_index)

    train_mask, val_mask, test_mask = build_masks(labels)

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.float32),
    )
    data = data.to(device)

    train_mask_t = torch.tensor(train_mask).to(device)
    val_mask_t = torch.tensor(val_mask).to(device)
    test_mask_t = torch.tensor(test_mask).to(device)

    # ── Class weight ──────────────────────────────────────────────────────────
    y_train = labels[train_mask]
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    logger.info(f"Positive class weight: {pos_weight.item():.2f}x")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GraphSAGEClassifier(
        in_channels=features.shape[1],
        hidden_channels=args.hidden_dim,
        embedding_dim=64,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    logger.info(
        f"Model — in:{features.shape[1]} hidden:{args.hidden_dim} emb:64 "
        f"dropout:{args.dropout}"
    )
    logger.debug(f"Architecture:\n{model}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment("aml_graphsage_graph")

    with mlflow.start_run(
        run_name=f"graphsage_lr{args.lr}_h{args.hidden_dim}_ep{args.epochs}"
    ):
        mlflow.log_params({
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": 64,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "n_nodes": len(features),
            "n_edges": edge_index.shape[1],
            "n_features": features.shape[1],
            "pos_weight": round(float(pos_weight.item()), 2),
            "seed": SEED,
        })

        best_val_auc = 0.0
        best_epoch = 0
        os.makedirs("models/graphsage", exist_ok=True)

        # ── Epoch loop ────────────────────────────────────────────────────────
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()

            optimizer.zero_grad()
            logits, _ = model(data.x, data.edge_index)

            loss = criterion(logits[train_mask_t], data.y[train_mask_t])

            # NaN guard
            if torch.isnan(loss):
                logger.error(f"NaN loss at epoch {epoch} — stopping")
                raise ValueError("NaN loss detected")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_time = time.time() - epoch_start

            if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
                val_metrics = evaluate(model, data, val_mask_t, device)
                scheduler.step(val_metrics["auc_pr"])

                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"loss: {loss.item():.4f} | "
                    f"val AUC-PR: {val_metrics['auc_pr']:.4f} | "
                    f"val F1: {val_metrics['f1_fraud']:.4f} | "
                    f"{epoch_time:.1f}s"
                )

                mlflow.log_metrics({
                    "train_loss": round(loss.item(), 4),
                    "val_auc_pr": val_metrics["auc_pr"],
                    "val_f1_fraud": val_metrics["f1_fraud"],
                    "val_recall_fraud": val_metrics["recall_fraud"],
                }, step=epoch)

                if val_metrics["auc_pr"] > best_val_auc:
                    best_val_auc = val_metrics["auc_pr"]
                    best_epoch = epoch
                    torch.save(model.state_dict(), "models/graphsage/graphsage_best.pt")
                    torch.save(
                        model.encoder.state_dict(),
                        "models/graphsage/graphsage_encoder.pt",
                    )
                    logger.info(
                        f"  -> New best val AUC-PR: {best_val_auc:.4f} - model saved"
                    )

        # ── Final evaluation ──────────────────────────────────────────────────
        logger.info(f"Loading best model from epoch {best_epoch}...")
        model.load_state_dict(
            torch.load("models/graphsage/graphsage_best.pt", map_location=device)
        )

        val_metrics = evaluate(model, data, val_mask_t, device)
        test_metrics = evaluate(model, data, test_mask_t, device)
        total_time = time.time() - train_start

        logger.info("=" * 60)
        logger.info(f"Training complete in {total_time/60:.1f} min")
        logger.info(f"Val  AUC-PR: {val_metrics['auc_pr']:.4f} | "
                    f"F1: {val_metrics['f1_fraud']:.4f} | "
                    f"Recall: {val_metrics['recall_fraud']:.4f}")
        logger.info(f"Test AUC-PR: {test_metrics['auc_pr']:.4f} | "
                    f"F1: {test_metrics['f1_fraud']:.4f} | "
                    f"Recall: {test_metrics['recall_fraud']:.4f}")
        logger.info("=" * 60)

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("training_time_min", round(total_time / 60, 2))
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_artifact("models/graphsage/graphsage_best.pt")
        mlflow.log_artifact("models/graphsage/graphsage_encoder.pt")

        # ── Save metrics JSON ─────────────────────────────────────────────────
        out = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
            "training_time_min": round(total_time / 60, 2),
            "hyperparams": {
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "embedding_dim": 64,
                "dropout": args.dropout,
                "epochs": args.epochs,
            },
        }
        with open("graphsage_metrics.json", "w") as f:
            json.dump(out, f, indent=2)
        mlflow.log_artifact("graphsage_metrics.json")
        logger.info("Metrics saved to graphsage_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Train GraphSAGE on Elliptic graph")
    p.add_argument(
        "--features",
        default="data/processed/graph_features.npy",
    )
    p.add_argument(
        "--edge_index",
        default="data/processed/graph_edge_index.npy",
    )
    p.add_argument(
        "--labels",
        default="data/processed/graph_labels.npy",
    )
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--max_nodes",
        type=int,
        default=None,
        help="Subsample to N nodes for smoke tests",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
