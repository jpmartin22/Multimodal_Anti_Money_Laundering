"""
Visualization utilities for Multimodal Anti Money Laundering.

This module contains functions for creating plots, charts, and other visualizations.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt


def plot_training_history(
    history: dict[str, list[float]], output_path: Optional[str] = None
) -> None:
    """Plot training history curves.

    Args:
        history: Dictionary containing training metrics.
        output_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric, values in history.items():
        ax.plot(values, label=metric)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_confusion_matrix(
    cm: Any, labels: Optional[list[str]] = None, output_path: Optional[str] = None
) -> None:
    """Plot a confusion matrix.

    Args:
        cm: Confusion matrix array.
        labels: Optional class labels.
        output_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(cm, cmap="Blues")

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")

    plt.colorbar(im, ax=ax)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
