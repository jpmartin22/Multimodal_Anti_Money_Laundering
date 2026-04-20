"""Project-wide configuration and path constants.

Access paths via the module-level constants so code does not depend
on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and training-run settings."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    early_stopping_patience: int = 10


@dataclass(frozen=True)
class DataConfig:
    """Data-split and preprocessing settings."""

    train_test_split: float = 0.8
    val_split: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class Config:
    """Top-level configuration composing sub-configs."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


DEFAULT_CONFIG = Config()
