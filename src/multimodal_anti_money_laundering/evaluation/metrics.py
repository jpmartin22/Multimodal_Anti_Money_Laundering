"""AML evaluation metrics.

Primary metric is AUC-PR (area under precision-recall curve) because
the ~2% illicit class makes ROC-AUC misleadingly optimistic.

Key metrics:
  - auc_pr          : AUC-PR (primary)
  - precision_at_r80: Precision when recall ≥ 0.80 (regulatory target)
  - fpr             : False positive rate at threshold that achieves recall=0.80
  - f1              : F1 at default 0.5 threshold
  - roc_auc         : Reported for completeness; not the primary target
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)


def compute_aml_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = 0.80,
) -> dict[str, float]:
    """Compute the full AML metric suite for a single model.

    Args:
        y_true: Binary ground-truth labels (0=licit, 1=illicit).
        y_score: Predicted illicit-class probabilities in [0, 1].
        target_recall: Recall level at which to measure precision and FPR.

    Returns:
        dict with auc_pr, precision_at_r80, fpr_at_r80, f1, roc_auc.
    """
    recall_tag = f"r{int(target_recall * 100)}"

    auc_pr = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))

    # precision_recall_curve returns one extra point; align with thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    valid = recalls[:-1] >= target_recall
    if valid.any():
        idx = int(np.where(valid)[0][-1])
        precision_at_target = float(precisions[idx])
        threshold_at_target = float(thresholds[idx])
        y_pred_t = (y_score >= threshold_at_target).astype(int)
        tn = int(((y_pred_t == 0) & (y_true == 0)).sum())
        fp = int(((y_pred_t == 1) & (y_true == 0)).sum())
        fpr_at_target = fp / max(tn + fp, 1)
    else:
        precision_at_target = 0.0
        fpr_at_target = 1.0

    y_pred_default = (y_score >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred_default, zero_division=0))

    metrics = {
        "auc_pr": round(auc_pr, 4),
        f"precision_at_{recall_tag}": round(precision_at_target, 4),
        f"fpr_at_{recall_tag}": round(fpr_at_target, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }
    logger.info("Metrics: %s", metrics)
    return metrics


def log_metrics_table(
    results: dict[str, dict[str, float]],
    title: str = "Model Comparison",
) -> None:
    """Print a side-by-side comparison table.

    Args:
        results: {model_name: metrics_dict} mapping.
        title: Table heading printed above the table.
    """
    if not results:
        return
    cols = list(next(iter(results.values())).keys())
    header = f"{'Model':<30} " + " ".join(f"{c:>20}" for c in cols)
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for name, m in results.items():
        row = f"{name:<30} " + " ".join(
            f"{m.get(c, float('nan')):>20.4f}" for c in cols
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for compatibility with existing tests)
# ---------------------------------------------------------------------------


def classification_report(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 as a dict."""
    return {
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "precision": float(
            sk_metrics.precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "recall": float(
            sk_metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(
            sk_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def regression_report(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MAE, MSE, RMSE, and R² as a dict."""
    mse = float(sk_metrics.mean_squared_error(y_true, y_pred))
    return {
        "mae": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(sk_metrics.r2_score(y_true, y_pred)),
    }
