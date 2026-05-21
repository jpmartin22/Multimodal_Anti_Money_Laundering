"""Debugging utilities for local and containerized environments.

Usage — drop a breakpoint anywhere:
    from multimodal_anti_money_laundering.utils.debug import set_trace
    set_trace()

Usage — remote debug from VS Code inside Docker:
    from multimodal_anti_money_laundering.utils.debug import attach_remote_debugger
    attach_remote_debugger()   # listens on 0.0.0.0:5678
    # Then attach in VS Code via the "Python: Remote Attach" launch config
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interactive breakpoint — prefers ipdb, falls back to pdb
# ---------------------------------------------------------------------------

def set_trace() -> None:
    """Drop into an interactive debugger at the call site.

    Uses ipdb when available (richer tab-completion, syntax highlighting).
    Falls back to stdlib pdb so this is safe in any environment.
    Respects the AML_DEBUG env var — no-op when AML_DEBUG is not set,
    so accidental calls in production are harmless.
    """
    if not os.getenv("AML_DEBUG"):
        logger.debug("set_trace() called but AML_DEBUG not set — skipping")
        return

    try:
        import ipdb  # type: ignore[import]
        ipdb.set_trace(context=10)
    except ImportError:
        import pdb
        pdb.set_trace()


# ---------------------------------------------------------------------------
# Remote debugger — VS Code / PyCharm attach inside Docker
# ---------------------------------------------------------------------------

def attach_remote_debugger(host: str = "0.0.0.0", port: int = 5678) -> None:
    """Start a debugpy server and block until a client attaches.

    Call this at the top of a training script when debugging inside Docker:

        docker compose run --service-ports debug-train

    Then in VS Code use the "AML Remote Debug" launch config (port 5678).

    The debugger is only started when AML_DEBUG=1 to prevent any effect
    in normal runs.
    """
    if not os.getenv("AML_DEBUG"):
        return

    import debugpy  # already in requirements.txt
    debugpy.listen((host, port))
    logger.info("debugpy listening on %s:%d — waiting for client...", host, port)
    debugpy.wait_for_client()
    logger.info("Debugger attached")


# ---------------------------------------------------------------------------
# Tensor / array inspection helpers
# ---------------------------------------------------------------------------

def assert_tensor_shape(
    tensor: torch.Tensor | np.ndarray,
    expected: tuple[int | None, ...],
    name: str = "tensor",
) -> None:
    """Assert tensor shape matches expected, with None as wildcard for any dim.

    Example:
        assert_tensor_shape(embeddings, (None, 128), "graphsage_emb")
        # passes for (64, 128) or (1024, 128); fails for (64, 64)
    """
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: rank mismatch — expected {len(expected)}D, got {len(actual)}D "
            f"(shape {actual})"
        )
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e is not None and a != e:
            raise AssertionError(
                f"{name}: dim {i} mismatch — expected {e}, got {a} (shape {actual})"
            )


def check_for_nans(tensor: torch.Tensor | np.ndarray, name: str = "tensor") -> bool:
    """Return True and log a warning if the tensor contains NaN or Inf values."""
    if isinstance(tensor, torch.Tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
    else:
        has_nan = bool(np.isnan(tensor).any())
        has_inf = bool(np.isinf(tensor).any())

    if has_nan:
        logger.warning("NaN detected in %s", name)
    if has_inf:
        logger.warning("Inf detected in %s", name)
    return bool(has_nan or has_inf)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def log_model_summary(model: Any, name: str = "model") -> None:
    """Log total and trainable parameter counts for a PyTorch model."""
    if not hasattr(model, "parameters"):
        logger.warning("log_model_summary: %s has no .parameters()", name)
        return

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "%s — total params: %s | trainable: %s",
        name,
        f"{total:,}",
        f"{trainable:,}",
    )


# ---------------------------------------------------------------------------
# Memory snapshot (Docker OOM helper)
# ---------------------------------------------------------------------------

def log_memory_usage(tag: str = "") -> None:
    """Log current process RSS and GPU memory (if available) for OOM diagnosis."""
    import os
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / 1024 ** 2
        logger.info("[mem%s] RSS: %.1f MB", f":{tag}" if tag else "", rss_mb)
    except ImportError:
        logger.debug("psutil not installed — skipping CPU memory log")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        logger.info(
            "[mem%s] GPU allocated: %.1f MB | reserved: %.1f MB",
            f":{tag}" if tag else "",
            alloc,
            reserved,
        )
