"""
train_bilstm.py
===============
Member B (Neha) — Week 2 BiLSTM training on Elliptic behavioral sequences.

Trains a 2-layer Bidirectional LSTM on the preprocessed windows from
elliptic_preprocess.py and logs everything to MLflow.

Architecture (matches proposal exactly)
----------------------------------------
    Input  : (batch, 30, 5)
    BiLSTM : 2 layers, hidden=64, bidirectional → output (batch, 128)
    Dropout: 0.3
    Linear : 128 → 64
    Output : 64-dimensional embedding  ← used by fusion head
    Head   : 64 → 1, sigmoid           ← used for standalone eval

Usage
-----
    # Full training (CPU, ~15 min as per proposal):
    python train_bilstm.py

    # Smoke test (fast, for CI):
    python train_bilstm.py --epochs 1 --max_samples 5000
"""

import argparse
import json
import os

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score,
                              precision_recall_curve,
                              classification_report)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from yaml import parser

# ─────────────────────────────────────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    """
    2-layer Bidirectional LSTM encoder.

    Input  : (batch, seq_len=30, n_features=5)
    Output : (batch, embedding_dim=64)  ← embedding for fusion head
    """
    def __init__(
        self,
        input_size:    int = 5,
        hidden_size:   int = 64,
        num_layers:    int = 2,
        dropout:       float = 0.3,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        # BiLSTM output is hidden_size * 2 (forward + backward)
        self.dropout    = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, (hn, _) = self.lstm(x)
        # hn: (num_layers * 2, batch, hidden_size)
        # Take last layer forward + backward hidden states
        forward_h  = hn[-2]   # (batch, hidden_size)
        backward_h = hn[-1]   # (batch, hidden_size)
        combined   = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden*2)
        embedding  = self.projection(self.dropout(combined))     # (batch, 64)
        return embedding


class BiLSTMClassifier(nn.Module):
    """
    Full classifier: BiLSTMEncoder + binary classification head.
    The encoder is extracted for the fusion head separately.
    """
    def __init__(self, input_size: int = 165, embedding_dim: int = 64):
        super().__init__()
        self.encoder = BiLSTMEncoder(input_size=input_size, embedding_dim=embedding_dim)
        self.head    = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        logit     = self.head(embedding)
        return logit.squeeze(-1), embedding


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    windows_path: str,
    labels_path:  str,
    max_samples:  int = None,
):
    X = np.load(windows_path)   # (N, 30, 5)
    y = np.load(labels_path)    # (N,)

    if max_samples and max_samples < len(X):
        # Stratified subsample
        idx_fraud  = np.where(y == 1)[0]
        idx_legit  = np.where(y == 0)[0]
        n_fraud    = min(len(idx_fraud), max(1, int(max_samples * y.mean())))
        n_legit    = max_samples - n_fraud
        idx        = np.concatenate([
            np.random.choice(idx_fraud, n_fraud, replace=False),
            np.random.choice(idx_legit, n_legit, replace=False),
        ])
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]

    print(f"\n  Loaded windows : {X.shape}")
    print(f"  Fraud rate     : {y.mean()*100:.3f}%  ({y.sum():,} / {len(y):,})")
    return X, y


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """BCELoss pos_weight to handle class imbalance."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    weight = n_neg / max(n_pos, 1)
    print(f"  Positive class weight: {weight:.1f}x")
    return torch.tensor([weight], dtype=torch.float32)


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            probs, _ = model(X_batch)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    auc_pr = average_precision_score(labels, probs)

    # Precision at recall = 0.8
    precision, recall, _ = precision_recall_curve(labels, probs)
    idx = np.searchsorted(recall[::-1], 0.8)
    prec_at_r80 = float(precision[::-1][idx]) if idx < len(precision) else 0.0

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    fpr = 1 - report.get("0", {}).get("recall", 1.0)

    return {
        "auc_pr":            round(auc_pr, 4),
        "prec_at_recall_80": round(prec_at_r80, 4),
        "f1_fraud":          round(report.get("1", {}).get("f1-score", 0), 4),
        "recall_fraud":      round(report.get("1", {}).get("recall", 0), 4),
        "precision_fraud":   round(report.get("1", {}).get("precision", 0), 4),
        "false_positive_rate": round(fpr, 4),
        "accuracy":          round(report.get("accuracy", 0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X, y = load_data(args.windows, args.labels, args.max_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    def make_loader(X_arr, y_arr, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(input_size=165, embedding_dim=64).to(device)
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion  = nn.BCELoss(weight=None)   # pos_weight applied via manual scaling
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, verbose=True
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_experiment("aml_bilstm_behavioral")

    with mlflow.start_run(run_name=f"bilstm_ep{args.epochs}"):
        mlflow.log_params({
            "model":         "BiLSTM",
            "hidden_size":   64,
            "num_layers":    2,
            "bidirectional": True,
            "embedding_dim": 64,
            "dropout":       0.3,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "lr":            args.lr,
            "window_size":   X.shape[1],
            "n_features":    X.shape[2],
            "train_rows":    len(X_train),
            "val_rows":      len(X_val),
            "test_rows":     len(X_test),
            "seed":          SEED,
        })

        best_auc_pr    = 0.0
        best_model_path = os.path.join(args.output_dir, "bilstm_best.pt")
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"\n{'='*55}")
        print(f"  Training BiLSTM for {args.epochs} epoch(s)...")
        print(f"{'='*55}")

        for epoch in range(1, args.epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                preds, _ = model(X_batch)

                # Manual pos_weight scaling for BCELoss
                weights  = torch.where(y_batch == 1, pos_weight, torch.ones_like(y_batch))
                loss     = (weights * nn.functional.binary_cross_entropy(
                    preds, y_batch, reduction="none")).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # ── Validate ──────────────────────────────────────────────────────
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step(val_metrics["auc_pr"])

            print(f"  Epoch {epoch:02d}/{args.epochs:02d} | "
                  f"loss={avg_loss:.4f} | "
                  f"val_auc_pr={val_metrics['auc_pr']:.4f} | "
                  f"val_f1={val_metrics['f1_fraud']:.4f}")

            mlflow.log_metrics({
                "train_loss":   avg_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }, step=epoch)

            # Save best model
            if val_metrics["auc_pr"] > best_auc_pr:
                best_auc_pr = val_metrics["auc_pr"]
                torch.save(model.state_dict(), best_model_path)

        # ── Final test evaluation ──────────────────────────────────────────────
        model.load_state_dict(torch.load(best_model_path))
        test_metrics = evaluate(model, test_loader, device)

        print(f"\n{'='*55}")
        print(f"  FINAL TEST METRICS")
        print(f"{'='*55}")
        for k, v in test_metrics.items():
            print(f"  {k:<25}: {v}")
        print(f"{'='*55}\n")

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # ── Save encoder separately for fusion head ────────────────────────────
        encoder_path = os.path.join(args.output_dir, "bilstm_encoder.pt")
        torch.save(model.encoder.state_dict(), encoder_path)
        print(f"  Encoder saved → {encoder_path}")
        print(f"  Full model saved → {best_model_path}")

        # ── Save metrics JSON for CI eval gate ────────────────────────────────
        metrics_path = "bilstm_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

        mlflow.log_artifact(best_model_path)
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(metrics_path)

        # ── Eval gate check ────────────────────────────────────────────────────
        print("  Evaluation gate check:")
        gate_passed = test_metrics["auc_pr"] >= 0.70   # lower bar for single branch
        print(f"  AUC-PR {test_metrics['auc_pr']:.4f} {'≥' if gate_passed else '<'} 0.70 "
              f"→ {'PASSED ✓' if gate_passed else 'FAILED ✗'}")

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Member B — BiLSTM training on PaySim behavioral windows."
    )
    parser.add_argument("--windows", type=str, default="data/processed/bilstm_sequences.npy")
    parser.add_argument("--labels",  type=str, default="data/processed/bilstm_labels.npy")
    parser.add_argument("--output_dir",  type=str, default="models/bilstm")
    parser.add_argument("--epochs",      type=int, default=10,
                        help="Training epochs (default: 10; use 1 for smoke test)")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset size for smoke tests (e.g. 5000)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

