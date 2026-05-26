"""ablation_study.py
===================
Member A (Jaya) — Phase 3 ablation study for the late-fusion model.

Trains the LateFusionMLP four times, each time zeroing out a different
subset of modalities, to measure each modality's contribution to AUC-PR.

Variants
--------
  graphsage_only    : BiLSTM=0, BERT=0
  bilstm_only       : GraphSAGE=0, BERT=0
  graphsage_bilstm  : BERT=0  (best no-text baseline)
  full_fusion       : all three modalities (BERT=0 until cache available)

Output
------
  reports/ablation_results.json  — AUC-PR + F1 per variant
  reports/ablation_results.png   — bar chart

Usage
-----
  python -m multimodal_anti_money_laundering.ablation_study
  python -m multimodal_anti_money_laundering.ablation_study --epochs 30 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from multimodal_anti_money_laundering.models.fusion import BERT_HIDDEN, LateFusionMLP
from multimodal_anti_money_laundering.train_bilstm import BiLSTMEncoder
from multimodal_anti_money_laundering.train_fusion import (
    BILSTM_ENCODER,
    BILSTM_SEQUENCES,
    DISTILBERT_CACHE,
    GRAPH_EDGE_INDEX,
    GRAPH_FEATURES,
    GRAPH_LABELS,
    GRAPHSAGE_ENCODER,
    extract_bilstm_embeddings,
    extract_graphsage_embeddings,
)
from multimodal_anti_money_laundering.train_graphsage import GraphSAGEEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

REPORTS_DIR = Path("reports")

VARIANTS = {
    "graphsage_only": {"use_gs": True, "use_bl": False, "use_bert": False},
    "bilstm_only": {"use_gs": False, "use_bl": True, "use_bert": False},
    "graphsage_bilstm": {"use_gs": True, "use_bl": True, "use_bert": False},
    "full_fusion": {"use_gs": True, "use_bl": True, "use_bert": True},
}


def train_variant(
    gs_emb: np.ndarray,
    bl_emb: np.ndarray,
    bert_cls: np.ndarray,
    labels: np.ndarray,
    use_gs: bool,
    use_bl: bool,
    use_bert: bool,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train one MLP variant and return test metrics."""
    n = len(labels)

    # Zero out inactive modalities
    gs = gs_emb if use_gs else np.zeros_like(gs_emb)
    bl = bl_emb if use_bl else np.zeros_like(bl_emb)
    bc = bert_cls if use_bert else np.zeros_like(bert_cls)

    idx = np.arange(n)
    idx_tv, idx_test = train_test_split(
        idx, test_size=0.15, stratify=labels, random_state=SEED
    )
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15 / 0.85, stratify=labels[idx_tv], random_state=SEED
    )

    def tensors(idx_split):
        return (
            torch.tensor(gs[idx_split], dtype=torch.float32).to(device),
            torch.tensor(bl[idx_split], dtype=torch.float32).to(device),
            torch.tensor(bc[idx_split], dtype=torch.float32).to(device),
            torch.tensor(labels[idx_split], dtype=torch.float32)
            .unsqueeze(1)
            .to(device),
        )

    gs_tr, bl_tr, bc_tr, y_tr = tensors(idx_train)
    gs_val, bl_val, bc_val, y_val = tensors(idx_val)
    gs_te, bl_te, bc_te, _ = tensors(idx_test)

    model = LateFusionMLP().to(device)
    pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(gs_tr, bl_tr, bc_tr), y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_probs = (
                    torch.sigmoid(model(gs_val, bl_val, bc_val)).cpu().numpy().ravel()
                )
            val_auc = average_precision_score(labels[idx_val], val_probs)
            scheduler.step(1 - val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(gs_te, bl_te, bc_te).cpu().numpy().ravel()
    test_probs = 1 / (1 + np.exp(-test_logits))
    test_auc = average_precision_score(labels[idx_test], test_probs)

    precision, recall, thresholds = precision_recall_curve(labels[idx_test], test_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_thr = float(thresholds[np.argmax(f1_scores[:-1])])
    test_preds = (test_probs >= best_thr).astype(int)
    tp = int(((test_preds == 1) & (labels[idx_test] == 1)).sum())
    fp = int(((test_preds == 1) & (labels[idx_test] == 0)).sum())
    fn = int(((test_preds == 0) & (labels[idx_test] == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Platt calibration (fit but not used in AUC — just for completeness)
    with torch.no_grad():
        val_logits_np = model(gs_val, bl_val, bc_val).cpu().numpy().ravel()
    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    platt.fit(val_logits_np.reshape(-1, 1), labels[idx_val])

    return {
        "auc_pr": round(float(test_auc), 4),
        "f1_fraud": round(float(f1), 4),
        "precision_fraud": round(float(prec), 4),
        "recall_fraud": round(float(rec), 4),
        "val_auc_pr": round(float(best_val_auc), 4),
        "best_threshold": round(float(best_thr), 4),
        "modalities": {
            "graphsage": use_gs,
            "bilstm": use_bl,
            "distilbert": use_bert,
        },
    }


def plot_results(results: dict, out_path: Path) -> None:
    labels_order = ["graphsage_only", "bilstm_only", "graphsage_bilstm", "full_fusion"]
    display = {
        "graphsage_only": "GraphSAGE\nonly",
        "bilstm_only": "BiLSTM\nonly",
        "graphsage_bilstm": "GraphSAGE\n+ BiLSTM",
        "full_fusion": "Full Fusion\n(all 3)",
    }
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    x = np.arange(len(labels_order))
    auc_vals = [results[v]["auc_pr"] for v in labels_order]
    f1_vals = [results[v]["f1_fraud"] for v in labels_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x - 0.2, auc_vals, 0.35, label="AUC-PR", color=colors, alpha=0.9)
    ax.bar(x + 0.2, f1_vals, 0.35, label="F1 (fraud)", color=colors, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([display[v] for v in labels_order], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("AML Fusion — Ablation Study (AUC-PR & F1 per modality)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="AUC-PR target")

    for bar, val in zip(bars, auc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Ablation bar chart saved -> %s", out_path)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load raw data ─────────────────────────────────────────────────────────
    for p in [GRAPH_FEATURES, GRAPH_LABELS, GRAPH_EDGE_INDEX, BILSTM_SEQUENCES]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run elliptic_preprocess.py first.")

    graph_X = np.load(GRAPH_FEATURES).astype(np.float32)
    labels = np.load(GRAPH_LABELS).astype(np.float32)
    edge_index = np.load(GRAPH_EDGE_INDEX)
    bilstm_X = np.load(BILSTM_SEQUENCES).astype(np.float32)

    # ── Extract embeddings once ───────────────────────────────────────────────
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

    gs_emb = extract_graphsage_embeddings(graph_X, edge_index, graphsage_enc, device)
    del graph_X, edge_index, graphsage_enc

    bl_emb = extract_bilstm_embeddings(bilstm_X, bilstm_enc, device)
    del bilstm_X, bilstm_enc

    if DISTILBERT_CACHE.exists():
        bert_cls = np.load(DISTILBERT_CACHE).astype(np.float32)
        logger.info("DistilBERT cache loaded: %s", bert_cls.shape)
    else:
        logger.warning(
            "distilbert_cls.npy not found — full_fusion variant uses zero BERT embeddings"
        )
        bert_cls = np.zeros((len(labels), BERT_HIDDEN), dtype=np.float32)

    # ── Run each variant ──────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, flags in VARIANTS.items():
        logger.info("=" * 55)
        logger.info("Ablation variant: %s", name)
        logger.info(
            "  GraphSAGE=%s  BiLSTM=%s  BERT=%s",
            flags["use_gs"],
            flags["use_bl"],
            flags["use_bert"],
        )
        t0 = time.time()
        metrics = train_variant(
            gs_emb=gs_emb,
            bl_emb=bl_emb,
            bert_cls=bert_cls,
            labels=labels,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            **flags,
        )
        elapsed = time.time() - t0
        logger.info(
            "  AUC-PR=%.4f  F1=%.4f  val_AUC-PR=%.4f  (%.1fs)",
            metrics["auc_pr"],
            metrics["f1_fraud"],
            metrics["val_auc_pr"],
            elapsed,
        )
        results[name] = metrics

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = REPORTS_DIR / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Ablation results saved -> %s", json_path)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Variant':<22} {'AUC-PR':>8} {'F1':>8} {'Recall':>8}")
    print("=" * 60)
    for name, m in results.items():
        print(
            f"  {name:<22} {m['auc_pr']:>8.4f} {m['f1_fraud']:>8.4f} {m['recall_fraud']:>8.4f}"
        )
    print("=" * 60)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(results, REPORTS_DIR / "ablation_results.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study for AML fusion model")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--smoke", action="store_true", help="5 epochs for quick test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        args.epochs = 5
    run(args)
