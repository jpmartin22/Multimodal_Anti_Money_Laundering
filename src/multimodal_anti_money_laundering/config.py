"""Project-wide configuration and path constants.

Access paths via module-level constants so code is independent of the
current working directory.  Typed dataclasses capture hyperparameters so
they can be logged to MLflow without manual dict construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Hyperparameter configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSAGEConfig:
    """GraphSAGE encoder hyperparameters (Branch 1)."""

    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    focal_gamma: float = 2.0
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 1024
    seed: int = 42


@dataclass(frozen=True)
class DistilBERTConfig:
    """DistilBERT text encoder hyperparameters (Branch 2)."""

    model_name: str = "distilbert-base-uncased"
    projection_dim: int = 64
    max_length: int = 64
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 32
    seed: int = 42


@dataclass(frozen=True)
class BiLSTMConfig:
    """BiLSTM time-series encoder hyperparameters (Branch 3)."""

    input_size: int = 5  # [amount, hour, day_of_week, tx_type, velocity]
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    window_days: int = 30
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 256
    seed: int = 42


@dataclass(frozen=True)
class FusionConfig:
    """Late-fusion MLP hyperparameters."""

    # Input dim = graphsage(128) + distilbert(64) + bilstm(64) = 256
    input_dim: int = 256
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.3
    epochs: int = 50
    lr: float = 5e-4
    batch_size: int = 256
    seed: int = 42


@dataclass(frozen=True)
class XGBBaselineConfig:
    """XGBoost baseline hyperparameters."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 30
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level training settings (legacy compat + defaults)."""

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    early_stopping_patience: int = 10


@dataclass(frozen=True)
class DataConfig:
    """Data-split and preprocessing settings."""

    train_frac: float = 0.70
    val_frac: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class Config:
    """Top-level config composing all sub-configs."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    graphsage: GraphSAGEConfig = field(default_factory=GraphSAGEConfig)
    distilbert: DistilBERTConfig = field(default_factory=DistilBERTConfig)
    bilstm: BiLSTMConfig = field(default_factory=BiLSTMConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    baseline: XGBBaselineConfig = field(default_factory=XGBBaselineConfig)


DEFAULT_CONFIG = Config()
