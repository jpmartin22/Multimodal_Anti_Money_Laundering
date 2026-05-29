"""shap_explainer.py
====================
Member A (Jaya) — Phase 3 SHAP explainability for the fusion model.

Generates SHAP force plots showing which embedding dimensions (by modality)
drive each AML prediction. Uses KernelExplainer on the fused embedding space.

Outputs
-------
  outputs/shap_force_plot.png    — force plot for a single suspicious transaction
  outputs/shap_summary_plot.png  — summary bar plot (mean |SHAP| per modality)
  reports/shap_force_plot.png    — copy committed to git for grading
  reports/shap_summary_plot.png  — copy committed to git for grading
  reports/shap_values.npy        — raw SHAP values for test set (for audit trail)

Usage
-----
  python -m multimodal_anti_money_laundering.models.shap_explainer
  python -m multimodal_anti_money_laundering.models.shap_explainer --n_samples 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from multimodal_anti_money_laundering.models.fusion import BERT_HIDDEN, LateFusionMLP
from multimodal_anti_money_laundering.train_bilstm import BiLSTMEncoder
from multimodal_anti_money_laundering.train_graphsage import GraphSAGEEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shap_explainer")

REPORTS_DIR = Path("reports")
OUTPUTS_DIR = Path("outputs")
GRAPHSAGE_ENCODER = Path("models/graphsage/graphsage_encoder.pt")
BILSTM_ENCODER = Path("models/bilstm/bilstm_encoder.pt")
DISTILBERT_DIR = Path("models/distilbert/memo_model")
DISTILBERT_CACHE = Path("data/processed/distilbert_cls.npy")
FUSION_PT = Path("models/fusion/fusion_mlp.pt")

SEED = 42
np.random.seed(SEED)


# ── Load embeddings ───────────────────────────────────────────────────────────


def load_fused_embeddings(device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data and extract fused embeddings from all encoders."""
    graph_X = np.load("data/processed/graph_features.npy").astype(np.float32)
    labels = np.load("data/processed/graph_labels.npy").astype(np.float32)
    bilstm_X = np.load("data/processed/bilstm_sequences.npy").astype(np.float32)
    edge_index = np.load("data/processed/graph_edge_index.npy")

    # GraphSAGE embeddings
    gs_enc = GraphSAGEEncoder()
    gs_enc.load_state_dict(
        torch.load(GRAPHSAGE_ENCODER, map_location=device, weights_only=True)
    )
    gs_enc.eval().to(device)

    with torch.no_grad():
        x = torch.tensor(graph_X, dtype=torch.float32).to(device)
        ei = torch.tensor(edge_index, dtype=torch.long).to(device)
        gs_emb = gs_enc(x, ei).cpu().numpy()

    # BiLSTM embeddings
    bl_enc = BiLSTMEncoder()
    bl_enc.load_state_dict(
        torch.load(BILSTM_ENCODER, map_location=device, weights_only=True)
    )
    bl_enc.eval().to(device)

    with torch.no_grad():
        seqs = torch.tensor(bilstm_X, dtype=torch.float32).to(device)
        bl_emb = bl_enc(seqs).cpu().numpy()

    # DistilBERT embeddings — use cache to avoid reloading 267MB model
    if DISTILBERT_CACHE.exists():
        bert_cls = np.load(DISTILBERT_CACHE).astype(np.float32)
        logger.info("DistilBERT embeddings loaded from cache: %s", bert_cls.shape)
    else:
        logger.warning(
            "distilbert_cls.npy cache not found — using zero embeddings for text branch"
        )
        bert_cls = np.zeros((len(labels), BERT_HIDDEN), dtype=np.float32)

    fused = np.concatenate([gs_emb, bl_emb, bert_cls], axis=1)
    logger.info(
        "Fused embeddings: %s  |  illicit=%.1f%%", fused.shape, labels.mean() * 100
    )
    return fused, labels


# ── SHAP wrapper ──────────────────────────────────────────────────────────────


def make_predict_fn(fusion: LateFusionMLP, device: torch.device):
    """Return a numpy → numpy predict function for KernelExplainer."""
    gs_end = 64
    bl_end = 128  # 64+64

    def predict(X: np.ndarray) -> np.ndarray:
        gs = torch.tensor(X[:, :gs_end], dtype=torch.float32).to(device)
        bl = torch.tensor(X[:, gs_end:bl_end], dtype=torch.float32).to(device)
        bc = torch.tensor(X[:, bl_end:], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = fusion(gs, bl, bc)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return probs

    return predict


# ── Feature names ─────────────────────────────────────────────────────────────


def make_feature_names(fused_dim: int) -> list[str]:
    names = []
    names += [f"GraphSAGE_{i}" for i in range(64)]
    names += [f"BiLSTM_{i}" for i in range(64)]
    names += [f"DistilBERT_{i}" for i in range(fused_dim - 128)]
    return names


# ── Main ──────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    device = torch.device("cpu")  # KernelExplainer is CPU-bound anyway

    fused, labels = load_fused_embeddings(device)
    feature_names = make_feature_names(fused.shape[1])

    # Load fusion model
    fusion = LateFusionMLP()
    fusion.load_state_dict(
        torch.load(FUSION_PT, map_location=device, weights_only=True)
    )
    fusion.eval()
    logger.info("Fusion MLP loaded from %s", FUSION_PT)

    predict_fn = make_predict_fn(fusion, device)

    # Background: balanced sample for KernelExplainer
    illicit_idx = np.where(labels == 1)[0]
    licit_idx = np.where(labels == 0)[0]
    n_bg = min(50, len(illicit_idx), len(licit_idx))
    bg_idx = np.concatenate(
        [
            np.random.choice(illicit_idx, n_bg, replace=False),
            np.random.choice(licit_idx, n_bg, replace=False),
        ]
    )
    background = fused[bg_idx]
    logger.info("SHAP background: %d samples", len(background))

    # Explain N_SAMPLES test instances (mix of illicit + licit)
    n = args.n_samples
    test_idx = np.concatenate(
        [
            illicit_idx[: n // 2],
            licit_idx[: n // 2],
        ]
    )
    test_X = fused[test_idx]

    logger.info(
        "Running KernelExplainer on %d samples (this takes a few minutes)...",
        len(test_X),
    )
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test_X, nsamples=100)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(REPORTS_DIR / "shap_values.npy", shap_values)
    logger.info("SHAP values saved to %s", REPORTS_DIR / "shap_values.npy")

    # ── Force plot for the most suspicious transaction ─────────────────────────
    probs = predict_fn(test_X)
    most_suspicious = int(np.argmax(probs))
    logger.info(
        "Force plot: sample index %d  score=%.4f  true_label=%d",
        most_suspicious,
        probs[most_suspicious],
        int(labels[test_idx[most_suspicious]]),
    )

    shap.force_plot(
        explainer.expected_value,
        shap_values[most_suspicious],
        test_X[most_suspicious],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    for dest in [
        OUTPUTS_DIR / "shap_force_plot.png",
        REPORTS_DIR / "shap_force_plot.png",
    ]:
        plt.savefig(dest, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Force plot saved to %s and %s", OUTPUTS_DIR, REPORTS_DIR)

    # ── Summary bar plot — mean |SHAP| per modality ───────────────────────────
    abs_shap = np.abs(shap_values)
    modality_importance = {
        "GraphSAGE": float(abs_shap[:, :64].mean()),
        "BiLSTM": float(abs_shap[:, 64:128].mean()),
        "DistilBERT": float(abs_shap[:, 128:].mean()),
    }
    logger.info("Modality importance: %s", modality_importance)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(modality_importance.keys())
    values = list(modality_importance.values())
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("AML Modality Importance (SHAP)")
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.0002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )
    plt.tight_layout()
    for dest in [
        OUTPUTS_DIR / "shap_summary_plot.png",
        REPORTS_DIR / "shap_summary_plot.png",
    ]:
        plt.savefig(dest, dpi=150)
    plt.close()
    logger.info("Summary plot saved to %s and %s", OUTPUTS_DIR, REPORTS_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP explainability for AML fusion model")
    p.add_argument("--n_samples", type=int, default=20, help="Test samples to explain")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
