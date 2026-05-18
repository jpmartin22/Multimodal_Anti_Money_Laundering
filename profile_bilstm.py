"""
profile_bilstm.py
=================
Member B (Neha) — Phase 2 §3 Profiling & Optimization

Profiles train_bilstm.py using:
  1. cProfile     — CPU time per function call
  2. memory_profiler — peak memory usage

Outputs
-------
  reports/profiling/bilstm_cprofile.txt   — top 20 slowest functions
  reports/profiling/bilstm_memory.txt     — memory usage over time
  reports/profiling/bilstm_benchmark.json — before/after optimization metrics

Usage
-----
    python profile_bilstm.py
"""

import cProfile
import io
import json
import logging
import os
import pstats
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from memory_profiler import memory_usage
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("profiler")

OUTPUT_DIR = "reports/profiling"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT MODEL FROM EXISTING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, "src/multimodal_anti_money_laundering")
from train_bilstm import BiLSTMClassifier  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# LOAD A SMALL SUBSET FOR PROFILING (fast, representative)
# ─────────────────────────────────────────────────────────────────────────────


def load_subset(n: int = 5000):
    logger.info(f"Loading {n:,} samples for profiling...")
    X = np.load("data/processed/bilstm_sequences.npy")
    y = np.load("data/processed/bilstm_labels.npy")

    idx_fraud = np.where(y == 1)[0]
    idx_legit = np.where(y == 0)[0]
    n_fraud = min(len(idx_fraud), int(n * 0.1))
    n_legit = n - n_fraud
    idx = np.concatenate(
        [
            np.random.choice(idx_fraud, n_fraud, replace=False),
            np.random.choice(idx_legit, n_legit, replace=False),
        ]
    )
    np.random.shuffle(idx)
    logger.info(f"Subset loaded — shape: {X[idx].shape}")
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION TO PROFILE
# ─────────────────────────────────────────────────────────────────────────────


def training_loop(X, y, hidden_size=64, batch_size=256, epochs=3):
    """Minimal training loop used for profiling — 3 epochs, small subset."""
    device = torch.device("cpu")

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = BiLSTMClassifier(input_size=X.shape[2], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor(
        [(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32
    )

    model.train()
    for _epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds, _ = model(X_batch)
            weights = torch.where(y_batch == 1, pos_weight, torch.ones_like(y_batch))
            loss = (
                weights
                * nn.functional.binary_cross_entropy(preds, y_batch, reduction="none")
            ).mean()
            loss.backward()
            optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────
# 1. cPROFILE
# ─────────────────────────────────────────────────────────────────────────────


def run_cprofile(X, y):
    logger.info("Running cProfile...")

    profiler = cProfile.Profile()
    profiler.enable()
    training_loop(X, y)
    profiler.disable()

    # Save top 20 functions by cumulative time
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    report = stream.getvalue()
    path = os.path.join(OUTPUT_DIR, "bilstm_cprofile.txt")
    with open(path, "w") as f:
        f.write("BiLSTM cProfile Report — Top 20 functions by cumulative time\n")
        f.write("=" * 65 + "\n\n")
        f.write(report)

    logger.info(f"cProfile report saved → {path}")

    # Print top 10 to terminal
    print("\n" + "=" * 65)
    print("  TOP 10 SLOWEST FUNCTIONS (cProfile)")
    print("=" * 65)
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.strip_dirs()
    stats2.sort_stats("cumulative")
    stats2.print_stats(10)
    print(stream2.getvalue())

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2. MEMORY PROFILING
# ─────────────────────────────────────────────────────────────────────────────


def run_memory_profile(X, y):
    logger.info("Running memory profiler...")

    mem_usage = memory_usage(
        (training_loop, (X, y), {}),
        interval=0.5,
        timeout=300,
        retval=False,
    )

    peak_mb = max(mem_usage)
    base_mb = min(mem_usage)
    delta_mb = peak_mb - base_mb

    report = (
        f"BiLSTM Memory Profile Report\n"
        f"{'=' * 45}\n"
        f"Base memory   : {base_mb:.1f} MB\n"
        f"Peak memory   : {peak_mb:.1f} MB\n"
        f"Memory delta  : {delta_mb:.1f} MB\n"
        f"Samples taken : {len(mem_usage)}\n\n"
        f"Memory over time (MB):\n"
        + "\n".join([f"  t={i*0.5:.1f}s : {m:.1f} MB" for i, m in enumerate(mem_usage)])
    )

    path = os.path.join(OUTPUT_DIR, "bilstm_memory.txt")
    with open(path, "w") as f:
        f.write(report)

    logger.info(f"Memory profile saved → {path}")
    print(f"\n{'='*45}")
    print("  MEMORY PROFILING RESULTS")
    print(f"{'='*45}")
    print(f"  Base memory  : {base_mb:.1f} MB")
    print(f"  Peak memory  : {peak_mb:.1f} MB")
    print(f"  Delta        : {delta_mb:.1f} MB")
    print(f"{'='*45}\n")

    return peak_mb, delta_mb


# ─────────────────────────────────────────────────────────────────────────────
# 3. BEFORE / AFTER BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────


def run_benchmark(X, y):
    """
    Measure training time before and after two optimizations:
      Optimization 1: num_workers=0 → already set (avoids macOS multiprocessing issues)
      Optimization 2: pin_memory=False → correct for CPU
      Optimization 3: torch.no_grad() in eval → already implemented
    We benchmark batch_size effect as a concrete measurable optimization.
    """
    logger.info("Running before/after benchmark...")
    results = {}

    # BEFORE — small batch size (less efficient)
    logger.info("  Benchmarking batch_size=64 (before)...")
    t0 = time.time()
    training_loop(X, y, batch_size=64, epochs=2)
    results["before"] = {
        "batch_size": 64,
        "time_seconds": round(time.time() - t0, 2),
        "description": "Small batch size — more gradient updates, slower per epoch",
    }

    # AFTER — larger batch size (more efficient on CPU)
    logger.info("  Benchmarking batch_size=256 (after)...")
    t0 = time.time()
    training_loop(X, y, batch_size=256, epochs=2)
    results["after"] = {
        "batch_size": 256,
        "time_seconds": round(time.time() - t0, 2),
        "description": "Larger batch size — fewer updates, faster per epoch on CPU",
    }

    speedup = results["before"]["time_seconds"] / max(
        results["after"]["time_seconds"], 0.01
    )
    results["speedup_x"] = round(speedup, 2)
    results["optimization_1"] = "Increased batch size from 64 → 256"
    results["optimization_2"] = (
        "torch.no_grad() in evaluation loop — avoids gradient computation"
    )

    path = os.path.join(OUTPUT_DIR, "bilstm_benchmark.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark saved → {path}")
    print(f"\n{'='*45}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*45}")
    print(f"  Before (batch=64)  : {results['before']['time_seconds']}s")
    print(f"  After  (batch=256) : {results['after']['time_seconds']}s")
    print(f"  Speedup            : {speedup:.2f}x faster")
    print(f"{'='*45}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    logger.info("=" * 55)
    logger.info("BiLSTM Profiling — Phase 2 §3")
    logger.info("=" * 55)

    np.random.seed(42)
    X, y = load_subset(n=5000)

    # 1. cProfile
    run_cprofile(X, y)

    # 2. Memory profiling
    run_memory_profile(X, y)

    # 3. Before / after benchmark
    run_benchmark(X, y)

    logger.info("All profiling complete.")
    logger.info(f"Reports saved to: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
