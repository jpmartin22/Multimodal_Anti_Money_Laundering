# API Reference

The package is importable as `multimodal_anti_money_laundering` after running `pip install -e .`.

## `multimodal_anti_money_laundering.config`

Project-wide path constants and typed config dataclasses.

```python
from multimodal_anti_money_laundering.config import (
    PROJECT_ROOT,
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    Config, TrainingConfig, DataConfig, DEFAULT_CONFIG,
)
```

Use these constants instead of hard-coded relative paths — they resolve against the repo root regardless of the current working directory.

## `multimodal_anti_money_laundering.logging_config`

```python
from multimodal_anti_money_laundering.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

## `multimodal_anti_money_laundering.data`

| Function | Purpose |
|---|---|
| `load_raw(filename)` | Read CSV from `data/raw/` |
| `load_processed(filename)` | Read CSV from `data/processed/` |
| `save_processed(df, filename)` | Write CSV to `data/processed/` |
| `process_data(input_dir, output_dir)` | Raw → processed pipeline |

CLI: `python -m multimodal_anti_money_laundering.data.make_dataset [--input PATH] [--output PATH]`

## `multimodal_anti_money_laundering.features`

```python
from multimodal_anti_money_laundering.features import build_features

df_features = build_features(df_processed)
```

## `multimodal_anti_money_laundering.models`

### `BaseModel` (abstract)

Abstract interface with `fit`, `predict`, `save`, `load`. Extend this for any new estimator.

### `Model`

Reference implementation scaffold. Serializes via `joblib`.

```python
from pathlib import Path
from multimodal_anti_money_laundering.models import Model

model = Model(config={"lr": 0.01})
# model.fit(X_train, y_train)
model.save(Path("models/model.joblib"))
reloaded = Model.load(Path("models/model.joblib"))
```

## `multimodal_anti_money_laundering.evaluation`

```python
from multimodal_anti_money_laundering.evaluation import classification_report, regression_report

metrics = classification_report(y_true, y_pred)
# -> {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
```

## `multimodal_anti_money_laundering.visualization`

```python
from multimodal_anti_money_laundering.visualization import plot_training_history, plot_confusion_matrix
```

## `multimodal_anti_money_laundering.utils`

```python
from multimodal_anti_money_laundering.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

```bash
python -m multimodal_anti_money_laundering.train_model --epochs 100 --batch-size 64
python -m multimodal_anti_money_laundering.predict_model --model-path models/model.joblib --input data/processed/test.csv
```

## Configuration

Defaults live in `multimodal_anti_money_laundering.config.DEFAULT_CONFIG`. Override via CLI flags on the training/prediction entrypoints.

---

**Multimodal Anti Money Laundering** · Version see `multimodal_anti_money_laundering.__version__`
