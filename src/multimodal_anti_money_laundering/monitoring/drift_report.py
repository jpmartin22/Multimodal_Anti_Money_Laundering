"""Evidently AI drift reports for all three AML modalities.

Proposal spec (Section 1.4):
  - Graph features   : PSI  (Population Stability Index)
  - Time-series      : KS   (Kolmogorov-Smirnov test)
  - Text / memo      : cosine-proxy via wasserstein on text statistics
                       (swap for EmbeddingsDriftMetric once DistilBERT is trained)

Reference split = first 80% of data (training distribution).
Current split   = last 20%         (simulated production window).

Each modality produces a standalone HTML report saved to reports/drift/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import ColumnDriftMetric, EmbeddingsDriftMetric
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/drift")


def _split_reference_current(df: pd.DataFrame, ref_frac: float = 0.8):
    """Split DataFrame into reference (training) and current (production) windows."""
    cut = int(len(df) * ref_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _save_report(report: Report, name: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{name}.html"
    report.save_html(str(out))
    logger.info("Drift report saved → %s", out)
    return out


# ---------------------------------------------------------------------------
# Modality 1 — Graph features (PSI)
# ---------------------------------------------------------------------------


def graph_drift_report(graph_features_path: Path) -> Path:
    """PSI drift report on Elliptic node features (166-dim)."""
    logger.info("Running graph feature drift report (PSI)...")

    features = np.load(graph_features_path)  # shape: (N_nodes, 166)
    cols = [f"feat_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=cols)

    reference, current = _split_reference_current(df)

    # PSI on every feature column
    psi_metrics = [ColumnDriftMetric(column_name=col, stattest="psi") for col in cols]
    report = Report(metrics=[DataDriftPreset(stattest="psi")] + psi_metrics)
    report.run(reference_data=reference, current_data=current)

    return _save_report(report, "graph_psi_drift")


# ---------------------------------------------------------------------------
# Modality 2 — Behavioral time-series (KS test)
# ---------------------------------------------------------------------------


def timeseries_drift_report(sequences_path: Path, labels_path: Path) -> Path:
    """KS drift report on BiLSTM window summary statistics."""
    logger.info("Running time-series drift report (KS)...")

    sequences = np.load(sequences_path)  # shape: (N_windows, seq_len, n_features)
    labels = np.load(labels_path)  # shape: (N_windows,)

    # Summarise each window to scalar stats — keeps Evidently in tabular mode
    df = pd.DataFrame(
        {
            "amount_mean": sequences[:, :, 0].mean(axis=1),
            "amount_std": sequences[:, :, 0].std(axis=1),
            "amount_max": sequences[:, :, 0].max(axis=1),
            "hour_mean": sequences[:, :, 1].mean(axis=1),
            "day_of_week_mode": sequences[:, :, 2].mean(axis=1),
            "tx_type_mean": sequences[:, :, 3].mean(axis=1),
            "velocity_mean": sequences[:, :, 4].mean(axis=1),
            "velocity_max": sequences[:, :, 4].max(axis=1),
            "label": labels,
        }
    )

    reference, current = _split_reference_current(df)

    feature_cols = [c for c in df.columns if c != "label"]
    ks_metrics = [
        ColumnDriftMetric(column_name=col, stattest="ks") for col in feature_cols
    ]
    report = Report(metrics=[DataDriftPreset(stattest="ks")] + ks_metrics)
    report.run(reference_data=reference, current_data=current)

    return _save_report(report, "timeseries_ks_drift")


# ---------------------------------------------------------------------------
# Modality 3 — Payment memo text (wasserstein proxy for cosine)
# ---------------------------------------------------------------------------


def _load_distilbert_embedding_model(model_dir: Path | None):
    if model_dir is None or not model_dir.exists():
        return None, None

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        bert_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        return tokenizer, bert_model.distilbert.eval()
    except Exception:
        logger.exception("DistilBERT embedding model load failed")
        return None, None


def _compute_text_embeddings(texts: pd.Series, tokenizer, bert_model) -> np.ndarray:
    import torch

    tokens = tokenizer(
        texts.tolist(),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    tokens.pop("token_type_ids", None)
    tokens = {k: v.to("cpu") for k, v in tokens.items()}

    with torch.no_grad():
        out = bert_model(**tokens)

    return out.last_hidden_state[:, 0, :].cpu().numpy()


def text_drift_report(memo_csv_path: Path) -> Path:
    """Text drift report with optional DistilBERT embedding drift when available."""
    logger.info("Running text drift report (memo text drift)...")

    df_raw = pd.read_csv(memo_csv_path)

    text_col = next(
        (
            c
            for c in df_raw.columns
            if c.lower() in ("memo", "text", "description", "memo_text")
        ),
        df_raw.columns[0],
    )
    texts = df_raw[text_col].fillna("").astype(str)

    tokenizer, bert_model = _load_distilbert_embedding_model(
        Path("models/distilbert/memo_model")
    )

    if tokenizer is not None and bert_model is not None:
        logger.info("DistilBERT embeddings available — using EmbeddingsDriftMetric")
        embeddings = _compute_text_embeddings(texts, tokenizer, bert_model)
        embed_cols = [f"memo_emb_{i}" for i in range(embeddings.shape[1])]

        df = pd.DataFrame(
            {
                "char_len": texts.str.len(),
                "word_count": texts.str.split().str.len(),
                "unique_words": texts.apply(lambda t: len(set(t.lower().split()))),
                "digit_ratio": texts.apply(
                    lambda t: sum(c.isdigit() for c in t) / max(len(t), 1)
                ),
                "upper_ratio": texts.apply(
                    lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
                ),
                **{
                    col: embeddings[:, i]
                    for i, col in enumerate(embed_cols)
                },
            }
        )

        reference, current = _split_reference_current(df)

        text_metrics = [
            ColumnDriftMetric(column_name=col, stattest="wasserstein")
            for col in ["char_len", "word_count", "unique_words", "digit_ratio", "upper_ratio"]
        ]
        report = Report(
            metrics=[DataDriftPreset(stattest="wasserstein")] + text_metrics + [
                EmbeddingsDriftMetric("memo_embeddings")
            ]
        )
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=ColumnMapping(
                embeddings={"memo_embeddings": embed_cols}
            ),
        )
        return _save_report(report, "text_embeddings_drift")

    logger.info(
        "DistilBERT embeddings unavailable — falling back to text statistic drift report"
    )
    df = pd.DataFrame(
        {
            "char_len": texts.str.len(),
            "word_count": texts.str.split().str.len(),
            "unique_words": texts.apply(lambda t: len(set(t.lower().split()))),
            "digit_ratio": texts.apply(
                lambda t: sum(c.isdigit() for c in t) / max(len(t), 1)
            ),
            "upper_ratio": texts.apply(
                lambda t: sum(c.isupper() for c in t) / max(len(t), 1)
            ),
        }
    )

    reference, current = _split_reference_current(df)

    text_metrics = [
        ColumnDriftMetric(column_name=col, stattest="wasserstein") for col in df.columns
    ]
    report = Report(metrics=[DataDriftPreset(stattest="wasserstein")] + text_metrics)
    report.run(reference_data=reference, current_data=current)

    return _save_report(report, "text_wasserstein_drift")


# ---------------------------------------------------------------------------
# Run all three modalities together
# ---------------------------------------------------------------------------


def run_all_drift_reports(
    graph_features_path: Path,
    sequences_path: Path,
    labels_path: Path,
    memo_csv_path: Path,
) -> dict[str, Path]:
    """Generate drift reports for all three modalities. Returns paths to HTML reports."""
    results = {}

    try:
        results["graph"] = graph_drift_report(graph_features_path)
    except Exception:
        logger.exception("Graph drift report failed")

    try:
        results["timeseries"] = timeseries_drift_report(sequences_path, labels_path)
    except Exception:
        logger.exception("Time-series drift report failed")

    try:
        results["text"] = text_drift_report(memo_csv_path)
    except Exception:
        logger.exception("Text drift report failed")

    return results
