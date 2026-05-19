"""Prometheus metrics exporter for AML model performance.

Reads metrics JSON files produced by training scripts and exposes them
as Prometheus gauges on the /metrics endpoint of the FastAPI app.

Metrics exposed:
  aml_model_auc_pr          - AUC-PR per model branch
  aml_model_fpr             - False positive rate per model branch
  aml_model_f1              - F1 score per model branch
  aml_model_precision       - Precision per model branch
  aml_model_recall          - Recall per model branch
  aml_api_predictions_total - Total prediction requests (counter)
  aml_api_latency_seconds   - Prediction latency histogram
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry — shared across the app
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry()

# Model performance gauges — labelled by branch (bilstm, graphsage, fusion)
auc_pr_gauge = Gauge(
    "aml_model_auc_pr",
    "AUC-PR score per model branch",
    ["branch"],
    registry=REGISTRY,
)
fpr_gauge = Gauge(
    "aml_model_fpr",
    "False positive rate per model branch",
    ["branch"],
    registry=REGISTRY,
)
f1_gauge = Gauge(
    "aml_model_f1",
    "F1 score (fraud class) per model branch",
    ["branch"],
    registry=REGISTRY,
)
precision_gauge = Gauge(
    "aml_model_precision",
    "Precision (fraud class) per model branch",
    ["branch"],
    registry=REGISTRY,
)
recall_gauge = Gauge(
    "aml_model_recall",
    "Recall (fraud class) per model branch",
    ["branch"],
    registry=REGISTRY,
)

# API request metrics
predictions_counter = Counter(
    "aml_api_predictions_total",
    "Total number of prediction requests",
    ["flagged"],
    registry=REGISTRY,
)
latency_histogram = Histogram(
    "aml_api_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Loaders — read training output JSON files
# ---------------------------------------------------------------------------

_METRICS_FILES = {
    "bilstm":    Path("bilstm_metrics.json"),
    "graphsage": Path("graphsage_metrics.json"),
    "baseline":  Path("models/baseline_metrics.json"),
}


def load_model_metrics() -> None:
    """Read all metrics JSON files and update Prometheus gauges."""
    for branch, path in _METRICS_FILES.items():
        if not path.exists():
            logger.warning("Metrics file not found, skipping: %s", path)
            continue
        try:
            with open(path) as f:
                m = json.load(f)

            auc_pr_gauge.labels(branch=branch).set(m.get("auc_pr", 0))
            fpr_gauge.labels(branch=branch).set(m.get("false_positive_rate", 0))
            f1_gauge.labels(branch=branch).set(m.get("f1_fraud", m.get("f1", 0)))
            precision_gauge.labels(branch=branch).set(
                m.get("precision_fraud", m.get("precision", 0))
            )
            recall_gauge.labels(branch=branch).set(
                m.get("recall_fraud", m.get("recall", 0))
            )
            logger.info("Loaded metrics for branch '%s': AUC-PR=%.4f", branch, m.get("auc_pr", 0))
        except Exception:
            logger.exception("Failed to load metrics from %s", path)


def get_metrics_output() -> bytes:
    """Return Prometheus text format for the /metrics endpoint."""
    return generate_latest(REGISTRY)
