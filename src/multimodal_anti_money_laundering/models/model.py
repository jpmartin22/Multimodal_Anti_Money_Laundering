"""Default model implementation.

Replace the stub methods with a real estimator (sklearn pipeline,
PyTorch module, etc.) while keeping the ``BaseModel`` contract.

Model persistence uses ``joblib`` (the sklearn convention). Only load
artifacts from trusted sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from multimodal_anti_money_laundering.models.base import BaseModel


class Model(BaseModel):
    """Reference model scaffold."""

    def fit(self, X: Any, y: Any) -> "Model":
        raise NotImplementedError("Implement Model.fit for your task")

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("Implement Model.predict for your task")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "Model":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
