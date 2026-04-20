"""Model training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_anti_money_laundering.config import DEFAULT_CONFIG, MODELS_DIR, PROCESSED_DATA_DIR
from multimodal_anti_money_laundering.logging_config import get_logger, setup_logging
from multimodal_anti_money_laundering.utils.seed import set_seed

logger = get_logger(__name__)


def train(data_path: Path, model_dir: Path, epochs: int, batch_size: int, lr: float) -> None:
    """Train the model and persist the fitted artifact to ``model_dir``.

    Fill in the training loop / estimator fit for your problem and
    call ``model.save(model_dir / "model.joblib")`` at the end.
    """
    logger.info("Training with data=%s epochs=%d bs=%d lr=%g", data_path, epochs, batch_size, lr)
    model_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """CLI entrypoint for model training."""
    cfg = DEFAULT_CONFIG.training
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    train(args.data_path, args.model_dir, args.epochs, args.batch_size, args.learning_rate)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
