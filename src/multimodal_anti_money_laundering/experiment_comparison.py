"""
experiment_comparison.py
========================
Member D (Rajani) — Phase 2 §4 Experiment Comparison

Runs 3 controlled experiments against the AML serving API, each with a
different risk threshold, and logs all metrics to MLflow.  Produces a
comparison table (CSV + printed) so results can be reviewed side-by-side.

Experiments
-----------
  Run 1 — Conservative  : threshold=0.30 (high sensitivity, more flags)
  Run 2 — Balanced      : threshold=0.50 (production default)
  Run 3 — Strict        : threshold=0.70 (low sensitivity, fewer flags)

Metrics logged per run
----------------------
  threshold, mean_ms, std_ms, p50_ms, p95_ms, p99_ms, max_ms,
  throughput_rps, flagged_rate, sla_passed (P95 < 200 ms)

Outputs
-------
  reports/experiments/serving_experiment_comparison.csv
  reports/experiments/serving_experiment_comparison.md
  MLflow experiment: "aml-serving-threshold-comparison"

Usage
-----
    python src/multimodal_anti_money_laundering/experiment_comparison.py
    python src/multimodal_anti_money_laundering/experiment_comparison.py --n-requests 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time

sys.path.insert(0, "src")

import mlflow
import numpy as np
from fastapi.testclient import TestClient

from multimodal_anti_money_laundering.serving.api import app

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("experiment_comparison")

OUTPUT_DIR = "reports/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENT_NAME = "aml-serving-threshold-comparison"

SAMPLE_PAYLOAD = {
    "transaction_id": "exp-tx-001",
    "graph": {"node_features": [0.1] * 166},
    "memo_text": "consulting services invoice Q1 wire transfer payment",
    "time_series": {
        "window": [[100.0, 14.0, 2.0, 1.0, 500.0]] * 10
    },
}

EXPERIMENTS = [
    {"name": "conservative", "threshold": 0.30,
     "description": "Low threshold — high sensitivity, catches more potential fraud"},
    {"name": "balanced",     "threshold": 0.50,
     "description": "Default production threshold — balanced precision/recall"},
    {"name": "strict",       "threshold": 0.70,
     "description": "High threshold — fewer flags, lower false-positive rate"},
]

SLA_TARGET_MS = 200.0


def run_requests(client: TestClient, threshold: float, n: int) -> dict:
    """Send N /predict requests with the given threshold override and collect metrics."""
    import multimodal_anti_money_laundering.serving.api as api_module
    original_threshold = api_module._THRESHOLD
    api_module._THRESHOLD = threshold

    # Warmup — not included in measurements
    for _ in range(min(10, n // 5)):
        client.post("/predict", json=SAMPLE_PAYLOAD)

    latencies = []
    flagged_count = 0

    for i in range(n):
        payload = {**SAMPLE_PAYLOAD, "transaction_id": f"exp-tx-{i:05d}"}
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload)
        latencies.append(time.perf_counter() - t0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("flagged"):
                flagged_count += 1

    api_module._THRESHOLD = original_threshold

    arr = np.array(latencies) * 1000  # → ms
    p95 = float(np.percentile(arr, 95))

    return {
        "mean_ms":        round(float(arr.mean()), 3),
        "std_ms":         round(float(arr.std()), 3),
        "min_ms":         round(float(arr.min()), 3),
        "p50_ms":         round(float(np.percentile(arr, 50)), 3),
        "p95_ms":         round(p95, 3),
        "p99_ms":         round(float(np.percentile(arr, 99)), 3),
        "max_ms":         round(float(arr.max()), 3),
        "throughput_rps": round(n / (arr.sum() / 1000), 2),
        "flagged_rate":   round(flagged_count / n, 4),
        "sla_passed":     p95 < SLA_TARGET_MS,
        "n_requests":     n,
    }


def run_all_experiments(n: int = 300) -> list[dict]:
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = TestClient(app)
    results = []

    for exp in EXPERIMENTS:
        logger.info(
            "Running experiment '%s' (threshold=%.2f, n=%d)...",
            exp["name"], exp["threshold"], n,
        )

        with mlflow.start_run(run_name=exp["name"]):
            mlflow.set_tag("description", exp["description"])
            mlflow.log_param("threshold",    exp["threshold"])
            mlflow.log_param("n_requests",   n)
            mlflow.log_param("sla_target_ms", SLA_TARGET_MS)

            metrics = run_requests(client, exp["threshold"], n)

            mlflow.log_metric("mean_latency_ms",   metrics["mean_ms"])
            mlflow.log_metric("std_latency_ms",    metrics["std_ms"])
            mlflow.log_metric("p50_latency_ms",    metrics["p50_ms"])
            mlflow.log_metric("p95_latency_ms",    metrics["p95_ms"])
            mlflow.log_metric("p99_latency_ms",    metrics["p99_ms"])
            mlflow.log_metric("max_latency_ms",    metrics["max_ms"])
            mlflow.log_metric("throughput_rps",    metrics["throughput_rps"])
            mlflow.log_metric("flagged_rate",      metrics["flagged_rate"])
            mlflow.log_metric("sla_passed",        int(metrics["sla_passed"]))

        row = {
            "experiment":     exp["name"],
            "threshold":      exp["threshold"],
            "description":    exp["description"],
            **metrics,
        }
        results.append(row)

        logger.info(
            "  P95=%.2f ms | throughput=%.1f req/s | flagged_rate=%.1f%% | SLA=%s",
            metrics["p95_ms"],
            metrics["throughput_rps"],
            metrics["flagged_rate"] * 100,
            "PASS" if metrics["sla_passed"] else "FAIL",
        )

    return results


def save_comparison_table(results: list[dict]) -> None:
    csv_path = os.path.join(OUTPUT_DIR, "serving_experiment_comparison.csv")
    md_path  = os.path.join(OUTPUT_DIR, "serving_experiment_comparison.md")

    columns = [
        "experiment", "threshold", "mean_ms", "std_ms",
        "p50_ms", "p95_ms", "p99_ms", "throughput_rps",
        "flagged_rate", "sla_passed",
    ]

    # CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV saved → %s", csv_path)

    # Markdown table
    header = "| " + " | ".join(columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows   = []
    for r in results:
        sla = "✓" if r["sla_passed"] else "✗"
        vals = [
            r["experiment"],
            str(r["threshold"]),
            f"{r['mean_ms']:.2f}",
            f"{r['std_ms']:.2f}",
            f"{r['p50_ms']:.2f}",
            f"{r['p95_ms']:.2f}",
            f"{r['p99_ms']:.2f}",
            f"{r['throughput_rps']:.1f}",
            f"{r['flagged_rate']*100:.1f}%",
            sla,
        ]
        rows.append("| " + " | ".join(vals) + " |")

    with open(md_path, "w") as f:
        f.write("# AML Serving API — Threshold Experiment Comparison\n\n")
        f.write(f"MLflow experiment: `{EXPERIMENT_NAME}`  \n")
        f.write(f"SLA target: P95 < {SLA_TARGET_MS} ms\n\n")
        f.write(header + "\n")
        f.write(sep    + "\n")
        f.write("\n".join(rows) + "\n")

    logger.info("Markdown saved → %s", md_path)

    # Print to console
    print("\n" + "=" * 70)
    print("AML Serving API — Experiment Comparison (3 runs)")
    print("=" * 70)
    col_w = [15, 10, 9, 9, 9, 9, 9, 15, 13, 10]
    hdrs  = ["Experiment", "Threshold", "Mean ms", "Std ms",
             "P50 ms", "P95 ms", "P99 ms", "Throughput", "Flagged %", "SLA"]
    print("  ".join(h.ljust(w) for h, w in zip(hdrs, col_w)))
    print("  ".join("-" * w for w in col_w))
    for r in results:
        sla = "PASS" if r["sla_passed"] else "FAIL"
        vals = [
            r["experiment"],
            str(r["threshold"]),
            f"{r['mean_ms']:.2f}",
            f"{r['std_ms']:.2f}",
            f"{r['p50_ms']:.2f}",
            f"{r['p95_ms']:.2f}",
            f"{r['p99_ms']:.2f}",
            f"{r['throughput_rps']:.1f} req/s",
            f"{r['flagged_rate']*100:.1f}%",
            sla,
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, col_w)))
    print("=" * 70)
    print(f"\nMLflow UI: mlflow ui --backend-store-uri mlruns\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3 serving API experiments and compare with MLflow"
    )
    parser.add_argument(
        "--n-requests", type=int, default=300,
        help="Requests per experiment run (default: 300)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AML Serving Experiment Comparison — Phase 2 §4")
    logger.info("Experiment: %s", EXPERIMENT_NAME)
    logger.info("=" * 60)

    results = run_all_experiments(n=args.n_requests)
    save_comparison_table(results)

    logger.info("=" * 60)
    logger.info("Done. MLflow runs recorded in mlruns/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
