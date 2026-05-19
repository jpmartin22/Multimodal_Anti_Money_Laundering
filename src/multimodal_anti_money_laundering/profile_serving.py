"""
profile_serving.py
==================
Member D (Rajani) — Phase 2 §3 Profiling & Optimization

Profiles the FastAPI serving/API component using:
  1. cProfile       — CPU time per function call during /predict requests
  2. memory_profiler — peak memory usage during sustained inference load
  3. Benchmark      — latency statistics: mean, P50, P95, P99 across N requests

Outputs
-------
  reports/profiling/serving_cprofile.txt   — top 30 slowest functions
  reports/profiling/serving_memory.txt     — memory usage over inference loop
  reports/profiling/serving_benchmark.json — latency + throughput benchmark

Usage
-----
    python src/multimodal_anti_money_laundering/profile_serving.py
    python src/multimodal_anti_money_laundering/profile_serving.py --n-requests 500
"""

import argparse
import cProfile
import io
import json
import logging
import os
import pstats
import sys
import time

sys.path.insert(0, "src")

import numpy as np
from fastapi.testclient import TestClient
from memory_profiler import memory_usage

from multimodal_anti_money_laundering.serving.api import app

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("serving_profiler")

OUTPUT_DIR = "reports/profiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic request payload — matches PredictRequest schema exactly
# ---------------------------------------------------------------------------
SAMPLE_PAYLOAD = {
    "transaction_id": "profile-tx-001",
    "graph": {"node_features": [0.1] * 166},
    "memo_text": "consulting services invoice Q1 wire transfer payment",
    "time_series": {
        "window": [[100.0, 14.0, 2.0, 1.0, 500.0]] * 10
    },
}


# ---------------------------------------------------------------------------
# 1. cProfile — CPU hotspots
# ---------------------------------------------------------------------------

def run_predict_n(client: TestClient, n: int) -> list[float]:
    """Run N predict requests and return list of latencies in seconds."""
    latencies = []
    for i in range(n):
        payload = {**SAMPLE_PAYLOAD, "transaction_id": f"profile-tx-{i:05d}"}
        t0 = time.perf_counter()
        client.post("/predict", json=payload)
        latencies.append(time.perf_counter() - t0)
    return latencies


def run_cprofile(n: int = 200) -> None:
    logger.info("Running cProfile over %d /predict requests...", n)
    client = TestClient(app)

    profiler = cProfile.Profile()
    profiler.enable()
    run_predict_n(client, n)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    out_path = os.path.join(OUTPUT_DIR, "serving_cprofile.txt")
    with open(out_path, "w") as f:
        f.write(f"cProfile — {n} /predict requests\n")
        f.write("=" * 60 + "\n")
        f.write(stream.getvalue())

    logger.info("cProfile output saved → %s", out_path)


# ---------------------------------------------------------------------------
# 2. memory_profiler — peak memory during inference
# ---------------------------------------------------------------------------

def _inference_loop(n: int = 100) -> None:
    """Target function for memory_profiler measurement."""
    client = TestClient(app)
    run_predict_n(client, n)


def run_memory_profile(n: int = 100) -> float:
    logger.info("Running memory_profiler over %d /predict requests...", n)

    mem_usage = memory_usage((_inference_loop, (n,), {}), interval=0.1, retval=False)

    peak_mb = max(mem_usage)
    baseline_mb = mem_usage[0]
    delta_mb = peak_mb - baseline_mb

    out_path = os.path.join(OUTPUT_DIR, "serving_memory.txt")
    with open(out_path, "w") as f:
        f.write(f"memory_profiler — {n} /predict requests\n")
        f.write("=" * 60 + "\n")
        f.write(f"Baseline memory : {baseline_mb:.1f} MB\n")
        f.write(f"Peak memory     : {peak_mb:.1f} MB\n")
        f.write(f"Delta (overhead): {delta_mb:.1f} MB\n\n")
        f.write("Full trace (MB over time):\n")
        for i, m in enumerate(mem_usage):
            f.write(f"  t={i*0.1:.1f}s  {m:.2f} MB\n")

    logger.info("Memory profile saved → %s", out_path)
    logger.info("Peak: %.1f MB | Delta: %.1f MB", peak_mb, delta_mb)
    return peak_mb


# ---------------------------------------------------------------------------
# 3. Benchmark — latency statistics
# ---------------------------------------------------------------------------

def run_benchmark(n: int = 500) -> dict:
    logger.info("Running latency benchmark over %d /predict requests...", n)
    client = TestClient(app)

    # Warmup
    for _ in range(10):
        client.post("/predict", json=SAMPLE_PAYLOAD)

    latencies = run_predict_n(client, n)
    arr = np.array(latencies) * 1000  # convert to ms

    results = {
        "n_requests":      n,
        "mean_ms":         round(float(arr.mean()), 3),
        "std_ms":          round(float(arr.std()), 3),
        "min_ms":          round(float(arr.min()), 3),
        "p50_ms":          round(float(np.percentile(arr, 50)), 3),
        "p95_ms":          round(float(np.percentile(arr, 95)), 3),
        "p99_ms":          round(float(np.percentile(arr, 99)), 3),
        "max_ms":          round(float(arr.max()), 3),
        "sla_target_ms":   200.0,
        "sla_passed":      bool(np.percentile(arr, 95) < 200.0),
        "throughput_rps":  round(n / (arr.sum() / 1000), 2),
    }

    out_path = os.path.join(OUTPUT_DIR, "serving_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark results:")
    logger.info("  Mean     : %.2f ms", results["mean_ms"])
    logger.info("  P50      : %.2f ms", results["p50_ms"])
    logger.info("  P95      : %.2f ms  (SLA target < 200 ms)", results["p95_ms"])
    logger.info("  P99      : %.2f ms", results["p99_ms"])
    logger.info("  Throughput: %.1f req/s", results["throughput_rps"])
    logger.info("  SLA PASSED: %s", results["sla_passed"])
    logger.info("Benchmark saved → %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the AML serving API")
    parser.add_argument("--n-requests", type=int, default=500,
                        help="Number of requests for benchmark (default: 500)")
    parser.add_argument("--n-profile",  type=int, default=200,
                        help="Number of requests for cProfile (default: 200)")
    parser.add_argument("--n-memory",   type=int, default=100,
                        help="Number of requests for memory profile (default: 100)")
    args = parser.parse_args()

    logger.info("=" * 55)
    logger.info("AML Serving API Profiler — Phase 2 §3")
    logger.info("=" * 55)

    run_cprofile(n=args.n_profile)
    run_memory_profile(n=args.n_memory)
    run_benchmark(n=args.n_requests)

    logger.info("=" * 55)
    logger.info("All profiling complete. Reports in %s/", OUTPUT_DIR)
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
