"""
train_graphsage_hydra.py
========================
Member A (Jaya) — Phase 2 §6 Configuration Management

Hydra-powered version of train_graphsage.py.
All hyperparameters are controlled via conf/ YAML files.

Config structure
----------------
    conf/
      config.yaml               <- entry point (composes defaults)
      model/
        graphsage_base.yaml     <- hidden=128, dropout=0.3 (AUC-PR 0.9261)
        graphsage_large.yaml    <- hidden=256, dropout=0.5 (AUC-PR 0.9299, best)
      data/
        elliptic.yaml           <- data paths + split ratios
      training/
        default.yaml            <- lr, epochs, grad_clip, mlflow settings
        fast.yaml               <- smoke-test config (5 epochs, 8k nodes)

Usage
-----
    # Default config (base model):
    python src/multimodal_anti_money_laundering/train_graphsage_hydra.py

    # Use large model config:
    python src/multimodal_anti_money_laundering/train_graphsage_hydra.py model=graphsage_large

    # Use fast config (CI smoke test):
    python src/multimodal_anti_money_laundering/train_graphsage_hydra.py training=fast

    # Override any value from command line:
    python src/multimodal_anti_money_laundering/train_graphsage_hydra.py training.lr=0.005 model.hidden_channels=64

    # Compose large model + fast training:
    python src/multimodal_anti_money_laundering/train_graphsage_hydra.py model=graphsage_large training=fast

    # Print resolved config without running:
    python src/multimodal_anti_money_laundering/train_graphsage_hydra.py --cfg job
"""

import json
import logging
import logging.handlers
import os
import sys
import time

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_graphsage import GraphSAGEClassifier  # reuse model definition

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("graphsage_hydra")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt_console = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
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


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────


def load_graph(cfg: DictConfig, logger: logging.Logger):
    logger.info(f"Loading graph — features: {cfg.data.features_path}")
    features   = np.load(cfg.data.features_path).astype(np.float32)
    edge_index = np.load(cfg.data.edge_index_path)
    labels     = np.load(cfg.data.labels_path)

    max_nodes = cfg.training.get("max_nodes", None)
    if max_nodes and max_nodes < len(features):
        idx_fraud = np.where(labels == 1)[0]
        idx_legit = np.where(labels == 0)[0]
        n_fraud = min(len(idx_fraud), max(1, int(max_nodes * labels.mean())))
        n_legit = max_nodes - n_fraud
        keep = np.concatenate([
            np.random.choice(idx_fraud, n_fraud, replace=False),
            np.random.choice(idx_legit, n_legit, replace=False),
        ])
        keep_set = set(keep.tolist())
        old_to_new = {old: new for new, old in enumerate(keep)}
        mask = np.array([
            s in keep_set and d in keep_set
            for s, d in zip(edge_index[0], edge_index[1])
        ])
        ei = edge_index[:, mask]
        ei = np.array([[old_to_new[v] for v in ei[0]], [old_to_new[v] for v in ei[1]]])
        features, labels, edge_index = features[keep], labels[keep], ei
        logger.info(f"Subsampled to {max_nodes:,} nodes")

    logger.info(
        f"Graph loaded — nodes: {len(features):,} | "
        f"edges: {edge_index.shape[1]:,} | "
        f"fraud: {labels.mean()*100:.2f}%"
    )
    return features, edge_index, labels


def build_masks(labels, cfg, logger):
    idx = np.arange(len(labels))
    idx_tv, idx_test = train_test_split(
        idx, test_size=cfg.data.test_size,
        stratify=labels, random_state=cfg.data.seed
    )
    val_frac = cfg.data.val_size / (1 - cfg.data.test_size)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_frac,
        stratify=labels[idx_tv], random_state=cfg.data.seed
    )
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask   = torch.zeros(len(labels), dtype=torch.bool)
    test_mask  = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val]     = True
    test_mask[idx_test]   = True
    logger.info(
        f"Split — Train: {train_mask.sum():,} | "
        f"Val: {val_mask.sum():,} | Test: {test_mask.sum():,}"
    )
    return train_mask, val_mask, test_mask, labels[idx_train]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x.to(device), data.edge_index.to(device))
        probs  = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = data.y[mask].cpu().numpy()

    auc_pr = average_precision_score(labels, probs)
    prec, recall, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * prec * recall / (prec + recall + 1e-8)
    best = np.argmax(f1s[:-1])
    thresh = float(thresholds[best])
    preds  = (probs >= thresh).astype(int)
    idx80  = np.searchsorted(recall[::-1], 0.8)
    p80    = float(prec[::-1][idx80]) if idx80 < len(prec) else 0.0
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "auc_pr"            : round(float(auc_pr), 4),
        "prec_at_recall_80" : round(p80, 4),
        "f1_fraud"          : round(report.get("1", {}).get("f1-score", 0), 4),
        "recall_fraud"      : round(report.get("1", {}).get("recall", 0), 4),
        "precision_fraud"   : round(report.get("1", {}).get("precision", 0), 4),
        "false_positive_rate": round(1 - report.get("0", {}).get("recall", 1.0), 4),
        "accuracy"          : round(report.get("accuracy", 0), 4),
        "optimal_threshold" : round(thresh, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN (Hydra entry point)
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Resolve working directory back to repo root (Hydra changes it)
    os.chdir(hydra.utils.get_original_cwd())

    logger = setup_logger(cfg.training.log_file)

    logger.info("=" * 60)
    logger.info("GraphSAGE — Hydra config")
    logger.info("=" * 60)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    np.random.seed(cfg.data.seed)
    torch.manual_seed(cfg.data.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    features, edge_index, labels = load_graph(cfg, logger)
    train_mask, val_mask, test_mask, y_train = build_masks(labels, cfg, logger)

    data = Data(
        x          = torch.tensor(features, dtype=torch.float32),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        y          = torch.tensor(labels, dtype=torch.float32),
    ).to(device)

    train_mask = train_mask.to(device)
    val_mask   = val_mask.to(device)
    test_mask  = test_mask.to(device)

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    logger.info(f"pos_weight: {pos_weight.item():.2f}x")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GraphSAGEClassifier(
        in_channels     = cfg.model.in_channels,
        hidden_channels = cfg.model.hidden_channels,
        embedding_dim   = cfg.model.embedding_dim,
        dropout         = cfg.model.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    logger.info(
        f"Model — in:{cfg.model.in_channels} "
        f"hidden:{cfg.model.hidden_channels} "
        f"emb:{cfg.model.embedding_dim} "
        f"dropout:{cfg.model.dropout}"
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment(cfg.training.mlflow_experiment)
    run_name = (
        f"{cfg.experiment_name}_"
        f"lr{cfg.training.lr}_"
        f"h{cfg.model.hidden_channels}_"
        f"ep{cfg.training.epochs}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        os.makedirs(cfg.training.model_dir, exist_ok=True)
        best_val_auc = 0.0
        best_epoch   = 0
        train_start  = time.time()

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(data.x, data.edge_index)
            loss = criterion(logits[train_mask], data.y[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.training.grad_clip
            )
            optimizer.step()

            if epoch % cfg.training.eval_every == 0 or epoch == 1:
                val_m = evaluate(model, data, val_mask, device)
                scheduler.step(val_m["auc_pr"])

                mlflow.log_metrics({
                    "train_loss"   : round(loss.item(), 4),
                    "val_auc_pr"   : val_m["auc_pr"],
                    "val_f1_fraud" : val_m["f1_fraud"],
                }, step=epoch)

                logger.info(
                    f"Epoch {epoch:3d}/{cfg.training.epochs} | "
                    f"loss: {loss.item():.4f} | "
                    f"val AUC-PR: {val_m['auc_pr']:.4f} | "
                    f"val F1: {val_m['f1_fraud']:.4f}"
                )

                if val_m["auc_pr"] > best_val_auc:
                    best_val_auc = val_m["auc_pr"]
                    best_epoch   = epoch
                    torch.save(
                        model.state_dict(),
                        f"{cfg.training.model_dir}/graphsage_best.pt"
                    )
                    torch.save(
                        model.encoder.state_dict(),
                        f"{cfg.training.model_dir}/graphsage_encoder.pt"
                    )
                    logger.info(f"  -> New best: {best_val_auc:.4f} - saved")

        # ── Final eval ────────────────────────────────────────────────────────
        model.load_state_dict(
            torch.load(
                f"{cfg.training.model_dir}/graphsage_best.pt", map_location=device
            )
        )
        val_metrics  = evaluate(model, data, val_mask, device)
        test_metrics = evaluate(model, data, test_mask, device)
        total_min    = (time.time() - train_start) / 60

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("training_time_min", round(total_min, 2))
        mlflow.log_metric("best_epoch", best_epoch)

        logger.info("=" * 60)
        logger.info(f"Training complete in {total_min:.1f} min | Best epoch: {best_epoch}")
        logger.info(
            f"Val  AUC-PR: {val_metrics['auc_pr']:.4f} | "
            f"F1: {val_metrics['f1_fraud']:.4f}"
        )
        logger.info(
            f"Test AUC-PR: {test_metrics['auc_pr']:.4f} | "
            f"F1: {test_metrics['f1_fraud']:.4f}"
        )
        logger.info("=" * 60)

        out = {
            "config"      : OmegaConf.to_container(cfg, resolve=True),
            "val_metrics" : val_metrics,
            "test_metrics": test_metrics,
            "best_epoch"  : best_epoch,
            "training_time_min": round(total_min, 2),
        }
        with open(cfg.training.metrics_output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Metrics saved to {cfg.training.metrics_output}")


if __name__ == "__main__":
    main()
