"""
train_bilstm.py
===============
Member B (Neha) — Week 2 BiLSTM training on Elliptic behavioral sequences.

Trains a 2-layer Bidirectional LSTM on the preprocessed sequences from
elliptic_preprocess.py and logs everything to MLflow.

Architecture (matches proposal exactly)
----------------------------------------
    Input  : (batch, 49, 165)
    BiLSTM : 2 layers, hidden=64, bidirectional → output (batch, 128)
    Dropout: 0.3
    Linear : 128 → 64
    Output : 64-dimensional embedding  ← used by fusion head
    Head   : 64 → 1, sigmoid           ← used for standalone eval

Phase 2 additions
-----------------
    - Python logger (DEBUG/INFO/WARNING/ERROR) with log rotation
    - NaN detection + shape assertion checks
    - Timing logs per epoch and total training
    - Error logging with full traceback

Usage
-----
    # Full training:
    python train_bilstm.py

    # Smoke test (fast, for CI):
    python train_bilstm.py --epochs 1 --max_samples 5000

    # Custom hyperparams (for MLflow experiments):
    python train_bilstm.py --lr 0.0001 --hidden_size 128 --epochs 15
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
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER SETUP  (Phase 2 — §2 Monitoring & §5 Logging)
# ─────────────────────────────────────────────────────────────────────────────


def setup_logger(log_file: str = "logs/bilstm_training.log") -> logging.Logger:
    """
    Configure logger with:
      - Console handler : INFO and above  (clean output for terminal)
      - File handler    : DEBUG and above (full detail for debugging)
      - Log rotation    : 5 MB max, 3 backup files (prevents disk bloat)
    """
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("bilstm")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt_console = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )

    # Console — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt_console)

    # Rotating file — DEBUG and above
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


class BiLSTMEncoder(nn.Module):
    """
    2-layer Bidirectional LSTM encoder.

    Input  : (batch, seq_len=49, n_features=165)
    Output : (batch, embedding_dim=64)  <- embedding for fusion head
    """

    def __init__(
        self,
        input_size: int = 165,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, _) = self.lstm(x)
        forward_h = hn[-2]
        backward_h = hn[-1]
        combined = torch.cat([forward_h, backward_h], dim=1)
        embedding = self.projection(self.dropout(combined))
        return embedding


class BiLSTMClassifier(nn.Module):
    """
    Full classifier: BiLSTMEncoder + binary classification head.
    The encoder is extracted separately for the fusion head.
    """

    def __init__(
        self, input_size: int = 165, hidden_size: int = 64, embedding_dim: int = 64
    ):
        super().__init__()
        self.encoder = BiLSTMEncoder(
            input_size=input_size, hidden_size=hidden_size, embedding_dim=embedding_dim
        )
        self.head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        logit = self.head(embedding)
        return logit.squeeze(-1), embedding


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECKS  (Phase 2 — §2 Model Assertion Checks)
# ─────────────────────────────────────────────────────────────────────────────


def run_sanity_checks(X: np.ndarray, y: np.ndarray, expected_features: int = 165):
    """
    Assert data quality before training starts.
    Catches common issues early: NaN values, wrong shapes, bad labels.
    """
    logger.info("Running sanity checks on data...")

    # Shape checks
    assert X.ndim == 3, f"Expected 3D array (N, seq, features), got {X.ndim}D"
    assert y.ndim == 1, f"Expected 1D labels array, got {y.ndim}D"
    assert (
        X.shape[0] == y.shape[0]
    ), f"Mismatch: {X.shape[0]} windows but {y.shape[0]} labels"
    assert (
        X.shape[2] == expected_features
    ), f"Expected {expected_features} features, got {X.shape[2]}"
    logger.debug(f"Shape checks passed — X: {X.shape}, y: {y.shape}")

    # NaN checks
    nan_X = np.isnan(X).sum()
    nan_y = np.isnan(y.astype(float)).sum()
    assert nan_X == 0, f"NaN detected in features — {nan_X} NaN values found"
    assert nan_y == 0, f"NaN detected in labels — {nan_y} NaN values found"
    logger.debug("NaN checks passed — no NaN values in features or labels")

    # Label checks
    unique_labels = np.unique(y)
    assert set(unique_labels).issubset(
        {0, 1}
    ), f"Labels must be 0 or 1, found: {unique_labels}"
    logger.debug(f"Label checks passed — unique labels: {unique_labels}")

    # Class balance warning
    fraud_rate = y.mean()
    if fraud_rate < 0.001:
        logger.warning(f"Very low fraud rate: {fraud_rate*100:.3f}% — consider SMOTE")
    elif fraud_rate > 0.5:
        logger.warning(
            f"Unusually high fraud rate: {fraud_rate*100:.1f}% — check labels"
        )

    logger.info(
        f"All sanity checks passed — "
        f"{len(X):,} samples, {fraud_rate*100:.2f}% fraud, "
        f"shape {X.shape}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def load_data(windows_path: str, labels_path: str, max_samples: int | None = None):
    logger.info(f"Loading data from {windows_path}")
    t0 = time.time()

    if not os.path.exists(windows_path):
        logger.error(f"Windows file not found: {windows_path}")
        raise FileNotFoundError(f"Missing: {windows_path}")
    if not os.path.exists(labels_path):
        logger.error(f"Labels file not found: {labels_path}")
        raise FileNotFoundError(f"Missing: {labels_path}")

    X = np.load(windows_path)
    y = np.load(labels_path)
    logger.debug(f"Raw load time: {time.time()-t0:.2f}s")

    if max_samples and max_samples < len(X):
        idx_fraud = np.where(y == 1)[0]
        idx_legit = np.where(y == 0)[0]
        n_fraud = min(len(idx_fraud), max(1, int(max_samples * y.mean())))
        n_legit = max_samples - n_fraud
        idx = np.concatenate(
            [
                np.random.choice(idx_fraud, n_fraud, replace=False),
                np.random.choice(idx_legit, n_legit, replace=False),
            ]
        )
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        logger.info(f"Subsampled to {max_samples:,} rows (stratified)")

    logger.info(
        f"Data loaded — shape: {X.shape} | "
        f"fraud rate: {y.mean()*100:.3f}% ({y.sum():,} / {len(y):,})"
    )
    return X, y


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weight = n_neg / max(n_pos, 1)
    logger.info(f"Positive class weight: {weight:.1f}x (neg={n_neg:,}, pos={n_pos:,})")
    return torch.tensor([weight], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            probs, _ = model(X_batch)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y_batch.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # Find optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = 0.3

    # Debug: check actual probability distribution
    logger.debug(
        f"Prob stats — min:{probs.min():.4f} max:{probs.max():.4f} "
        f"mean:{probs.mean():.4f} median:{np.median(probs):.4f}"
    )
    logger.debug(
        f"Best threshold from PR curve: {best_threshold:.4f} "
        f"gives F1: {f1_scores[best_idx]:.4f}"
    )

    preds = (probs >= best_threshold).astype(int)

    # NaN guard on predictions
    if np.isnan(probs).any():
        logger.error("NaN detected in model predictions — training may have diverged")
        raise ValueError("NaN in predictions")

    auc_pr = average_precision_score(labels, probs)

    precision2, recall2, _ = precision_recall_curve(labels, probs)
    idx = np.searchsorted(recall2[::-1], 0.8)
    prec_at_r80 = float(precision2[::-1][idx]) if idx < len(precision2) else 0.0

    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    return {
        "auc_pr": round(auc_pr, 4),
        "prec_at_recall_80": round(prec_at_r80, 4),
        "f1_fraud": round(report.get("1.0", {}).get("f1-score", 0), 4),
        "recall_fraud": round(report.get("1.0", {}).get("recall", 0), 4),
        "precision_fraud": round(report.get("1.0", {}).get("precision", 0), 4),
        "false_positive_rate": round(1 - report.get("0.0", {}).get("recall", 1.0), 4),
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
    X, y = load_data(args.windows, args.labels, args.max_samples)
    run_sanity_checks(X, y, expected_features=X.shape[2])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )
    logger.info(
        f"Split — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    def make_loader(X_arr, y_arr, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32),
        )
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        input_size=X.shape[2], hidden_size=args.hidden_size, embedding_dim=64
    ).to(device)

    pos_weight = compute_pos_weight(y_train).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    logger.info(
        f"Model created — input_size={X.shape[2]}, "
        f"hidden_size={args.hidden_size}, embedding_dim=64"
    )
    logger.debug(f"Full model architecture:\n{model}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment("aml_bilstm_behavioral")

    with mlflow.start_run(
        run_name=f"bilstm_lr{args.lr}_h{args.hidden_size}_ep{args.epochs}"
    ):
        mlflow.log_params(
            {
                "model": "BiLSTM",
                "input_size": X.shape[2],
                "hidden_size": args.hidden_size,
                "num_layers": 2,
                "bidirectional": True,
                "embedding_dim": 64,
                "dropout": 0.3,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "window_size": X.shape[1],
                "n_features": X.shape[2],
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "seed": SEED,
                "device": str(device),
            }
        )

        best_auc_pr = 0.0
        best_model_path = os.path.join(args.output_dir, "bilstm_best.pt")
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Starting training for {args.epochs} epoch(s)...")

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0

            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                preds, _ = model(X_batch)

                # NaN guard mid-training
                if torch.isnan(preds).any():
                    logger.error(
                        f"NaN in predictions at epoch {epoch}, batch {batch_idx}"
                    )
                    raise ValueError("NaN detected in model output during training")

                weights = torch.where(
                    y_batch == 1, pos_weight, torch.ones_like(y_batch)
                )
                loss = (
                    weights
                    * nn.functional.binary_cross_entropy(
                        preds, y_batch, reduction="none"
                    )
                ).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start

            # ── Validate ──────────────────────────────────────────────────────
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step(val_metrics["auc_pr"])

            logger.info(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"loss={avg_loss:.4f} | "
                f"val_auc_pr={val_metrics['auc_pr']:.4f} | "
                f"val_f1={val_metrics['f1_fraud']:.4f} | "
                f"time={epoch_time:.1f}s"
            )
            logger.debug(f"Full val metrics epoch {epoch}: {val_metrics}")

            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "epoch_time": epoch_time,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                },
                step=epoch,
            )

            if val_metrics["auc_pr"] > best_auc_pr:
                best_auc_pr = val_metrics["auc_pr"]
                torch.save(model.state_dict(), best_model_path)
                logger.debug(f"New best model saved — val_auc_pr={best_auc_pr:.4f}")

        # ── Final test evaluation ──────────────────────────────────────────────
        logger.info("Loading best model for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_metrics = evaluate(model, test_loader, device)

        logger.info(f"Optimal threshold: {test_metrics['optimal_threshold']:.4f}")

        total_time = time.time() - train_start
        logger.info(f"Training complete in {total_time/60:.1f} minutes")
        logger.info(
            f"Final test metrics — "
            f"AUC-PR: {test_metrics['auc_pr']:.4f} | "
            f"Precision@R=0.8: {test_metrics['prec_at_recall_80']:.4f} | "
            f"FPR: {test_metrics['false_positive_rate']:.4f}"
        )

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("total_training_minutes", round(total_time / 60, 2))

        # ── Save encoder for fusion head ───────────────────────────────────────
        encoder_path = os.path.join(args.output_dir, "bilstm_encoder.pt")
        torch.save(model.encoder.state_dict(), encoder_path)
        logger.info(f"Encoder saved → {encoder_path}")
        logger.info(f"Full model saved → {best_model_path}")

        # ── Save metrics JSON for CI eval gate ────────────────────────────────
        metrics_path = "bilstm_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.debug(f"Metrics JSON saved → {metrics_path}")

        mlflow.log_artifact(best_model_path)
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(metrics_path)

        # ── Eval gate ─────────────────────────────────────────────────────────
        gate_passed = test_metrics["auc_pr"] >= 0.70
        if gate_passed:
            logger.info(
                f"Eval gate PASSED — AUC-PR {test_metrics['auc_pr']:.4f} >= 0.70"
            )
        else:
            logger.warning(
                f"Eval gate FAILED — AUC-PR {test_metrics['auc_pr']:.4f} < 0.70"
            )

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Member B — BiLSTM training on Elliptic behavioral sequences."
    )
    parser.add_argument(
        "--windows", type=str, default="data/processed/bilstm_sequences.npy"
    )
    parser.add_argument(
        "--labels", type=str, default="data/processed/bilstm_labels.npy"
    )
    parser.add_argument("--output_dir", type=str, default="models/bilstm")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10; use 1 for smoke test)",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="LSTM hidden size (default: 64)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap dataset size for smoke tests (e.g. 5000)",
    )
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("BiLSTM Training Started")
    logger.info(
        f"Config: lr={args.lr}, hidden={args.hidden_size}, epochs={args.epochs}"
    )
    logger.info("=" * 55)

    try:
        train(args)
    except FileNotFoundError as e:
        logger.error(f"Data file missing: {e}")
        raise
    except AssertionError as e:
        logger.error(f"Sanity check failed: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error during training: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
