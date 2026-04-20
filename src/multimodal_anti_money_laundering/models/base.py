"""Abstract base class for models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    """Interface every model implementation must satisfy."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModel":
        """Fit the model to training data."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Return predictions for ``X``."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load a previously saved model from disk."""
