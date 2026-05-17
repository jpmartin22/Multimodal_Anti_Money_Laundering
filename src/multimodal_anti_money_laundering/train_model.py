"""Model training entrypoint.

Supports two modes via --model flag:
  baseline  — XGBoost on Elliptic tabular features (default, runs without GPU)
  graphsage — GraphSAGE on the transaction graph (requires PyG + GPU for speed)

Results are logged to MLflow and saved to models/.
Run `mlflow ui` in the project root to browse experiment runs.

Examples:
    python -m multimodal_anti_money_laundering.train_model --model baseline
    python -m multimodal_anti_money_laundering.train_model --model baseline --synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_anti_money_laundering.config import DEFAULT_CONFIG, MODELS_DIR
from multimodal_anti_money_laundering.logging_config import get_logger, setup_logging
from multimodal_anti_money_laundering.utils.seed import set_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def train_baseline(
    synthetic: bool = False, model_dir: Path = MODELS_DIR
) -> dict[str, float]:
    """Train XGBoost baseline and return test metrics.

    Args:
        synthetic: Use synthetic Elliptic-like data instead of real CSVs.
        model_dir: Where to save the fitted model.

    Returns:
        Dict with auc_pr, precision_at_r80, fpr_at_r80, f1, roc_auc.
    """
    try:
        import mlflow

        mlflow_available = True
    except ImportError:
        mlflow_available = False
        logger.warning("MLflow not installed — skipping experiment tracking.")

    cfg = DEFAULT_CONFIG.baseline
    set_seed(cfg.seed)

    from multimodal_anti_money_laundering.data.elliptic import load_tabular
    from multimodal_anti_money_laundering.models.baseline import XGBBaseline

    logger.info("Loading Elliptic tabular data (synthetic=%s) …", synthetic)
    X_train, y_train, X_val, y_val, X_test, y_test = load_tabular(
        use_synthetic=synthetic
    )

    model = XGBBaseline(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        seed=cfg.seed,
    )

    run_params = {
        "model": "xgb_baseline",
        "n_estimators": cfg.n_estimators,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "seed": cfg.seed,
        "synthetic": synthetic,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "illicit_frac_train": float(y_train.mean()),
    }

    if mlflow_available:
        mlflow.set_experiment("aml-baseline")
        with mlflow.start_run(run_name="xgb_baseline"):
            mlflow.log_params(run_params)
            model.fit(X_train, y_train, X_val, y_val, cfg.early_stopping_rounds)
            metrics = model.evaluate(X_test, y_test)
            mlflow.log_metrics(metrics)
            model_path = model_dir / "baseline_xgb.joblib"
            model.save(model_path)
            mlflow.log_artifact(str(model_path))
            logger.info("MLflow run logged. Launch `mlflow ui` to browse.")
    else:
        model.fit(X_train, y_train, X_val, y_val, cfg.early_stopping_rounds)
        metrics = model.evaluate(X_test, y_test)
        model_path = model_dir / "baseline_xgb.joblib"
        model.save(model_path)

    logger.info("Baseline test metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(description="Train AML models")
    parser.add_argument(
        "--model",
        choices=["baseline", "graphsage"],
        default="baseline",
        help="Which model to train (default: baseline)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory to save trained artifacts",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic Elliptic data (no Kaggle download required)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.training.seed,
    )
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "baseline":
        metrics = train_baseline(synthetic=args.synthetic, model_dir=args.model_dir)
    elif args.model == "graphsage":
        raise NotImplementedError("GraphSAGE training will be added in Week 2.")
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Write metrics JSON alongside model artifacts
    metrics_path = args.model_dir / f"{args.model}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
