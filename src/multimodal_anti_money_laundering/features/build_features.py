"""Feature engineering transformations."""

from __future__ import annotations

import pandas as pd

from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from a processed dataframe.

    Keep transformations deterministic and side-effect free so the
    function can be reused at inference time.
    """
    logger.info("Building features for %d rows", len(df))
    return df.copy()
