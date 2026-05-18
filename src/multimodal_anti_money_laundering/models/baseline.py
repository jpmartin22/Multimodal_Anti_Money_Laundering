"""XGBoost tabular baseline for the Elliptic dataset.

Trains on the 166 node features (no graph structure).  This establishes the
benchmark that GraphSAGE and the late-fusion model must beat on AUC-PR.

Usage:
    from multimodal_anti_money_laundering.models.baseline import XGBBaseline
    model = XGBBaseline()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from multimodal_anti_money_laundering.logging_config import get_logger

logger = get_logger(__name__)


class XGBBaseline:
    """XGBoost classifier tuned for imbalanced AML detection.

    ``scale_pos_weight`` is set automatically from training labels to handle
    the ~2% illicit class.  Focal-loss approximation via ``eval_metric='aucpr'``
    guides early stopping toward AUC-PR rather than accuracy.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self._model: Any = None

    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int = 30,
    ) -> XGBBaseline:
        """Train XGBoost with automatic class-weight balancing.

        Args:
            X_train: Feature matrix (n_samples, n_features).
            y_train: Binary labels (0=licit, 1=illicit).
            X_val: Optional validation features for early stopping.
            y_val: Optional validation labels for early stopping.
            early_stopping_rounds: Stop if val AUC-PR hasn't improved.

        Returns:
            self
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("XGBoost is required: pip install xgboost") from e

        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        logger.info(
            "Training XGBoost baseline | n=%d illicit=%d (%.2f%%) scale_pos_weight=%.1f",
            len(y_train),
            n_pos,
            100.0 * n_pos / len(y_train),
            scale_pos_weight,
        )

        # early_stopping_rounds is a constructor param in XGBoost ≥ 2.0
        early_stop = early_stopping_rounds if (X_val is not None) else None
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=early_stop,
            random_state=self.seed,
            n_jobs=-1,
        )

        self._model = xgb.XGBClassifier(**params)

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = 50

        self._model.fit(X_train, y_train, **fit_kwargs)
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return illicit-class probabilities, shape (n,)."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at *threshold*."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute AML-relevant metrics on a held-out split.

        Returns dict with keys: auc_pr, precision_at_r80, fpr, f1, roc_auc.
        """
        from multimodal_anti_money_laundering.evaluation.metrics import (
            compute_aml_metrics,
        )

        proba = self.predict_proba(X)
        return compute_aml_metrics(y, proba)

    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the fitted model to *path* using joblib."""
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved baseline model to %s", path)

    @classmethod
    def load(cls, path: Path) -> XGBBaseline:
        """Load a previously saved baseline from *path*."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected XGBBaseline, got {type(obj).__name__}")
        return obj
