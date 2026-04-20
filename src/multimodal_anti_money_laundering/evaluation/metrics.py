"""Metric helpers that return plain dicts for easy logging."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics as sk_metrics


def classification_report(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 as a dict."""
    return {
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "precision": float(
            sk_metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            sk_metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(sk_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def regression_report(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MAE, MSE, RMSE, and R^2 as a dict."""
    mse = float(sk_metrics.mean_squared_error(y_true, y_pred))
    return {
        "mae": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(sk_metrics.r2_score(y_true, y_pred)),
    }
