"""Raw-to-processed data pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_anti_money_laundering.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from multimodal_anti_money_laundering.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def process_data(input_dir: Path, output_dir: Path) -> None:
    """Transform raw data in ``input_dir`` and write outputs to ``output_dir``.

    Replace this stub with project-specific cleaning, feature extraction,
    and train/val/test splits.
    """
    logger.info("Reading raw data from %s", input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing processed data to %s", output_dir)


def main() -> None:
    """CLI entrypoint for data processing."""
    parser = argparse.ArgumentParser(description="Process raw data into model inputs")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    setup_logging()
    process_data(args.input, args.output)
    logger.info("Data processing complete")


if __name__ == "__main__":
    main()
