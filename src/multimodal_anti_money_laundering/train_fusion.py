"""train_fusion.py
=================
Member A (Jaya) — Phase 3 late-fusion MLP training.

Loads the three frozen encoders trained in Phase 2, extracts embeddings for
all labeled Elliptic nodes, trains the LateFusionMLP head, applies Platt
calibration, and logs everything to MLflow.

Architecture recap
------------------
  GraphSAGE  encoder  : 165 → 64  (frozen, models/graphsage/graphsage_encoder.pt)
  BiLSTM     encoder  : (49, 165) → 64  (frozen, models/bilstm/bilstm_encoder.pt)
  DistilBERT [CLS]    : memo_text → 768  (frozen, models/distilbert/memo_model/)
  LateFusionMLP       : 192 → 128 → 64 → 1  (trained here)
  Platt calibration   : sklearn CalibratedClassifierCV on val logits

Outputs
-------
  models/fusion/fusion_mlp.pt            — trained fusion head state dict
  models/fusion/fusion_calibrator.joblib — Platt calibrator
  fusion_metrics.json                    — test AUC-PR, F1, FPR

Usage
-----
  python -m multimodal_anti_money_laundering.train_fusion
  python -m multimodal_anti_money_laundering.train_fusion --epochs 30 --lr 0.0005
  python -m multimodal_anti_money_laundering.train_fusion --smoke  # 5 epochs, fast
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

from multimodal_anti_money_laundering.models.fusion import BERT_HIDDEN, LateFusionMLP
from multimodal_anti_money_laundering.train_bilstm import BiLSTMEncoder
from multimodal_anti_money_laundering.train_graphsage import GraphSAGEEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fusion")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Default paths ─────────────────────────────────────────────────────────────
GRAPH_FEATURES = Path("data/processed/graph_features.npy")
GRAPH_LABELS = Path("data/processed/graph_labels.npy")
GRAPH_EDGE_INDEX = Path("data/processed/graph_edge_index.npy")
BILSTM_SEQUENCES = Path("data/processed/bilstm_sequences.npy")
GRAPHSAGE_ENCODER = Path("models/graphsage/graphsage_encoder.pt")
BILSTM_ENCODER = Path("models/bilstm/bilstm_encoder.pt")
DISTILBERT_DIR = Path("models/distilbert/memo_model")
FUSION_OUT_DIR = Path("models/fusion")


# ── Synthetic memo generator ──────────────────────────────────────────────────

_ILLICIT_TEMPLATES = [
    "urgent wire transfer consulting services offshore account",
    "immediate payment advisory fee shell company",
    "confidential transaction business consulting urgent",
    "offshore investment services immediate transfer required",
    "anonymous wire payment urgent advisory services",
]
_LICIT_TEMPLATES = [
    "invoice payment monthly subscription services",
    "vendor payment supply chain logistics standard",
    "employee payroll direct deposit regular",
    "utility bill payment electricity standard monthly",
    "software license renewal annual payment",
]


def generate_memo(label: int, node_idx: int) -> str:
    templates = _ILLICIT_TEMPLATES if label == 1 else _LICIT_TEMPLATES
    return templates[node_idx % len(templates)]


# ── Embedding extraction ──────────────────────────────────────────────────────


@torch.no_grad()
def extract_graphsage_embeddings(
    features: np.ndarray,
    edge_index: np.ndarray,
    encoder: GraphSAGEEncoder,
    device: torch.device,
) -> np.ndarray:
    logger.info("Extracting GraphSAGE embeddings for %d nodes...", len(features))
    x = torch.tensor(features, dtype=torch.float32).to(device)
    ei = torch.tensor(edge_index, dtype=torch.long).to(device)
    emb = encoder(x, ei).cpu().numpy()
    logger.info("GraphSAGE embeddings: %s", emb.shape)
    return emb


@torch.no_grad()
def extract_bilstm_embeddings(
    sequences: np.ndarray,
    encoder: BiLSTMEncoder,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    logger.info("Extracting BiLSTM embeddings for %d sequences...", len(sequences))
    parts = []
    for start in range(0, len(sequences), batch_size):
        batch = torch.tensor(
            sequences[start : start + batch_size], dtype=torch.float32
        ).to(device)
        parts.append(encoder(batch).cpu().numpy())
    emb = np.concatenate(parts, axis=0)
    logger.info("BiLSTM embeddings: %s", emb.shape)
    return emb


DISTILBERT_CACHE = Path("data/processed/distilbert_cls.npy")


def extract_distilbert_embeddings(
    labels: np.ndarray, distilbert_dir: Path, device: torch.device, batch_size: int = 32
) -> np.ndarray:
    # Use cached embeddings if available (avoids loading 267MB model alongside other data)
    if DISTILBERT_CACHE.exists():
        emb = np.load(DISTILBERT_CACHE).astype(np.float32)
        logger.info("DistilBERT embeddings loaded from cache: %s", emb.shape)
        return emb

    if not distilbert_dir.exists():
        logger.warning("DistilBERT dir not found — using zero embeddings for text branch")
        return np.zeros((len(labels), BERT_HIDDEN), dtype=np.float32)

    import traceback

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(distilbert_dir))
        bert_model = AutoModelForSequenceClassification.from_pretrained(str(distilbert_dir))
        bert_base = bert_model.distilbert.eval().to(device)
        del bert_model  # free classifier head memory

        memos = [generate_memo(int(lbl), i) for i, lbl in enumerate(labels)]
        parts = []
        logger.info("Extracting DistilBERT [CLS] embeddings for %d memos...", len(memos))
        for start in range(0, len(memos), batch_size):
            batch_memos = memos[start : start + batch_size]
            tokens = tokenizer(
                batch_memos,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64,
            )
            tokens.pop("token_type_ids", None)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                out = bert_base(**tokens)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            parts.append(cls)
            if (start // batch_size) % 100 == 0:
                logger.info("  batch %d/%d", start // batch_size, len(memos) // batch_size)

        emb = np.concatenate(parts, axis=0)
        DISTILBERT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(DISTILBERT_CACHE, emb)
        logger.info("DistilBERT [CLS] embeddings saved to cache: %s  shape=%s", DISTILBERT_CACHE, emb.shape)
        return emb
    except Exception:
        print("DistilBERT extraction error:\n", traceback.format_exc())
        logger.exception("DistilBERT extraction failed — using zero embeddings")
        return np.zeros((len(labels), BERT_HIDDEN), dtype=np.float32)


def run_extract_bert_only(labels: np.ndarray) -> None:
    """Extract DistilBERT embeddings only and save to cache. Run this separately
    with a clean Python process before running the full fusion training."""
    device = torch.device("cpu")
    logger.info("Extracting DistilBERT embeddings (standalone mode)...")
    emb = extract_distilbert_embeddings(labels, DISTILBERT_DIR, device)
    logger.info("Done. Saved to %s  shape=%s", DISTILBERT_CACHE, emb.shape)


# ── Training ──────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load preprocessed data ────────────────────────────────────────────────
    for path in [GRAPH_FEATURES, GRAPH_LABELS, GRAPH_EDGE_INDEX, BILSTM_SEQUENCES]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run elliptic_preprocess.py first:\n"
                "  python -m multimodal_anti_money_laundering.elliptic_preprocess"
            )

    graph_X = np.load(GRAPH_FEATURES).astype(np.float32)
    labels = np.load(GRAPH_LABELS).astype(np.float32)
    edge_index = np.load(GRAPH_EDGE_INDEX)
    bilstm_X = np.load(BILSTM_SEQUENCES).astype(np.float32)

    logger.info(
        "Graph features: %s  |  Labels: %s  |  Illicit: %.1f%%",
        graph_X.shape,
        labels.shape,
        labels.mean() * 100,
    )

    # ── Load encoders (frozen) ────────────────────────────────────────────────
    graphsage_enc = GraphSAGEEncoder()
    graphsage_enc.load_state_dict(
        torch.load(GRAPHSAGE_ENCODER, map_location=device, weights_only=True)
    )
    graphsage_enc.eval().to(device)

    bilstm_enc = BiLSTMEncoder()
    bilstm_enc.load_state_dict(
        torch.load(BILSTM_ENCODER, map_location=device, weights_only=True)
    )
    bilstm_enc.eval().to(device)

    # ── Extract all embeddings ────────────────────────────────────────────────
    gs_emb = extract_graphsage_embeddings(graph_X, edge_index, graphsage_enc, device)
    del graph_X, edge_index, graphsage_enc  # free ~30 MB

    bl_emb = extract_bilstm_embeddings(bilstm_X, bilstm_enc, device)
    del bilstm_X, bilstm_enc  # free ~1.5 GB before loading DistilBERT

    if args.no_distilbert:
        logger.info("--no-distilbert: skipping text branch, using zero embeddings")
        bert_cls = np.zeros((len(labels), BERT_HIDDEN), dtype=np.float32)
    else:
        bert_cls = extract_distilbert_embeddings(labels, DISTILBERT_DIR, device)

    logger.info(
        "Embeddings — graph:%s  bilstm:%s  text:%s", gs_emb.shape, bl_emb.shape, bert_cls.shape
    )

    # ── Train / val / test split ──────────────────────────────────────────────
    idx = np.arange(len(labels))
    idx_tv, idx_test = train_test_split(
        idx, test_size=0.15, stratify=labels, random_state=SEED
    )
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15 / 0.85, stratify=labels[idx_tv], random_state=SEED
    )
    logger.info(
        "Split — train:%d  val:%d  test:%d", len(idx_train), len(idx_val), len(idx_test)
    )

    def to_tensors(idx_split):
        gs = torch.tensor(gs_emb[idx_split], dtype=torch.float32).to(device)
        bl = torch.tensor(bl_emb[idx_split], dtype=torch.float32).to(device)
        bc = torch.tensor(bert_cls[idx_split], dtype=torch.float32).to(device)
        y = torch.tensor(labels[idx_split], dtype=torch.float32).unsqueeze(1).to(device)
        return gs, bl, bc, y

    gs_tr, bl_tr, bc_tr, y_tr = to_tensors(idx_train)
    gs_val, bl_val, bc_val, y_val = to_tensors(idx_val)
    gs_te, bl_te, bc_te, y_te = to_tensors(idx_test)

    # ── Model + loss + optimizer ──────────────────────────────────────────────
    model = LateFusionMLP().to(device)
    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    logger.info(
        "pos_weight=%.2f  lr=%s  epochs=%d", pos_weight.item(), args.lr, args.epochs
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_experiment("AML-Fusion")
    with mlflow.start_run(run_name=f"fusion_ep{args.epochs}_lr{args.lr}"):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "lr": args.lr,
                "dropout": 0.3,
                "architecture": "192→128→64→1",
                "encoders": "graphsage+bilstm+distilbert",
                "calibration": "platt",
            }
        )

        best_val_auc = 0.0
        best_state = None
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            optimizer.zero_grad()
            logits = model(gs_tr, bl_tr, bc_tr)
            loss = criterion(logits, y_tr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validate every 5 epochs
            if epoch % 5 == 0 or epoch == args.epochs:
                model.eval()
                with torch.no_grad():
                    val_logits = model(gs_val, bl_val, bc_val)
                    val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
                val_auc = average_precision_score(labels[idx_val], val_probs)
                scheduler.step(1 - val_auc)

                logger.info(
                    "Epoch %3d/%d  loss=%.4f  val_AUC-PR=%.4f",
                    epoch,
                    args.epochs,
                    loss.item(),
                    val_auc,
                )
                mlflow.log_metrics(
                    {"train_loss": loss.item(), "val_auc_pr": val_auc}, step=epoch
                )

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        logger.info(
            "Training done in %.1fs  |  best val AUC-PR=%.4f",
            time.time() - t0,
            best_val_auc,
        )

        # ── Test evaluation ───────────────────────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_logits = model(gs_te, bl_te, bc_te).cpu().numpy().ravel()
        test_probs = 1 / (1 + np.exp(-test_logits))
        test_auc = average_precision_score(labels[idx_test], test_probs)

        precision, recall, thresholds = precision_recall_curve(
            labels[idx_test], test_probs
        )
        idx_r80 = np.searchsorted(recall[::-1], 0.8)
        prec_at_r80 = (
            float(precision[::-1][idx_r80]) if idx_r80 < len(precision) else 0.0
        )

        # Optimal F1 threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_thr = float(thresholds[np.argmax(f1_scores[:-1])])
        test_preds = (test_probs >= best_thr).astype(int)
        report = classification_report(
            labels[idx_test], test_preds, output_dict=True, zero_division=0
        )
        fpr = report.get("0", {}).get("recall", 1.0)
        fpr = 1 - fpr  # FPR = 1 - TNR

        logger.info(
            "Test  AUC-PR=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f  FPR=%.4f  threshold=%.3f",
            test_auc,
            report.get("1", {}).get("f1-score", 0),
            report.get("1", {}).get("precision", 0),
            report.get("1", {}).get("recall", 0),
            fpr,
            best_thr,
        )
        mlflow.log_metrics(
            {
                "test_auc_pr": test_auc,
                "test_f1": report.get("1", {}).get("f1-score", 0),
                "test_precision": report.get("1", {}).get("precision", 0),
                "test_recall": report.get("1", {}).get("recall", 0),
                "test_fpr": fpr,
                "best_threshold": best_thr,
            }
        )

        # ── Platt calibration ─────────────────────────────────────────────────
        logger.info("Fitting Platt calibration on val logits...")
        model.eval()
        with torch.no_grad():
            val_logits_np = model(gs_val, bl_val, bc_val).cpu().numpy().ravel()
        platt = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        platt.fit(val_logits_np.reshape(-1, 1), labels[idx_val])
        logger.info("Platt calibration fitted on %d val samples", len(idx_val))

        # ── Save artifacts ────────────────────────────────────────────────────
        FUSION_OUT_DIR.mkdir(parents=True, exist_ok=True)
        fusion_pt = FUSION_OUT_DIR / "fusion_mlp.pt"
        calib_path = FUSION_OUT_DIR / "fusion_calibrator.joblib"

        torch.save(model.state_dict(), fusion_pt)
        joblib.dump(platt, calib_path)

        metrics = {
            "auc_pr": round(test_auc, 4),
            "f1_fraud": round(report.get("1", {}).get("f1-score", 0), 4),
            "precision_fraud": round(report.get("1", {}).get("precision", 0), 4),
            "recall_fraud": round(report.get("1", {}).get("recall", 0), 4),
            "false_positive_rate": round(fpr, 4),
            "prec_at_recall_80": round(prec_at_r80, 4),
            "best_threshold": round(best_thr, 4),
            "val_auc_pr": round(best_val_auc, 4),
        }
        with open("fusion_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(fusion_pt))
        mlflow.log_artifact(str(calib_path))
        mlflow.log_artifact("fusion_metrics.json")

        logger.info("Saved: %s  %s  fusion_metrics.json", fusion_pt, calib_path)
        logger.info("Final test AUC-PR: %.4f", test_auc)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AML late-fusion MLP")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--smoke", action="store_true", help="5 epochs fast test")
    p.add_argument(
        "--no-distilbert",
        dest="no_distilbert",
        action="store_true",
        help="Skip DistilBERT extraction (use zero text embeddings) to save RAM",
    )
    p.add_argument(
        "--extract-bert-only",
        dest="extract_bert_only",
        action="store_true",
        help="Only extract DistilBERT embeddings and save cache, then exit",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.extract_bert_only:
        labels = np.load(GRAPH_LABELS).astype(np.float32)
        run_extract_bert_only(labels)
    else:
        if args.smoke:
            args.epochs = 5
            args.lr = 0.001
        train(args)
