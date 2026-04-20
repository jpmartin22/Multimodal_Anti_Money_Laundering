"""Dataset loading utilities.

Wrappers around pandas/numpy I/O that resolve paths against the
project's configured data directories.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from multimodal_anti_money_laundering.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)


def load_raw(filename: str) -> pd.DataFrame:
    """Load a CSV from the `data/raw` directory."""
    path = RAW_DATA_DIR / filename
    logger.info("Loading raw data: %s", path)
    return pd.read_csv(path)


def load_processed(filename: str) -> pd.DataFrame:
    """Load a CSV from the `data/processed` directory."""
    path = PROCESSED_DATA_DIR / filename
    logger.info("Loading processed data: %s", path)
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Write a dataframe to `data/processed`, creating the directory if needed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved processed data: %s", path)
    return path
