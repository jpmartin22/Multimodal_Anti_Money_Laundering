"""Model inference entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_anti_money_laundering.config import MODELS_DIR, PROCESSED_DATA_DIR
from multimodal_anti_money_laundering.logging_config import get_logger, setup_logging
from multimodal_anti_money_laundering.models.model import Model

logger = get_logger(__name__)


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    """Load a trained model and write predictions for ``input_path`` to ``output_path``."""
    logger.info("Loading model from %s", model_path)
    model = Model.load(model_path)

    logger.info("Scoring %s", input_path)
    _ = model  # replace with: df = pd.read_csv(input_path); preds = model.predict(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing predictions to %s", output_path)


def main() -> None:
    """CLI entrypoint for batch prediction."""
    parser = argparse.ArgumentParser(description="Generate predictions from a trained model")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "model.joblib")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "test.csv")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    setup_logging()
    predict(args.model_path, args.input, args.output)
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
