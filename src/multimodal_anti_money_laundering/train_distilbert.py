"""Hydra-powered DistilBERT fine-tuning for synthetic AML memo text.

Outputs
-------
  checkpoints/distilbert/memo_model/  — saved model + tokenizer
  distilbert_metrics.json             — eval metrics for DVC/CI gates
  mlruns/                             — MLflow experiment tracking folder

Requirements
------------
  pip install transformers datasets scikit-learn mlflow torch

Run
---
  # Default config:
  python src/multimodal_anti_money_laundering/train_distilbert.py

  # CPU smoke test:
  python src/multimodal_anti_money_laundering/train_distilbert.py training=distilbert_fast

  # Override individual values:
  python src/multimodal_anti_money_laundering/train_distilbert.py \
      training.epochs=1 data.max_samples=5000
"""

import json
import os
from collections.abc import Mapping
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL_NAMES = ["legitimate", "illicit"]

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────


def load_data(csv_path: str, max_samples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["memo_text"] = df["memo_text"].fillna("").astype(str)

    # Blank memos → placeholder token (model learns to ignore this)
    df["memo_text"] = df["memo_text"].replace("", "[NO_MEMO]")

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)

    print(
        f"Loaded {len(df):,} rows  |  "
        f"illicit: {df['is_illicit'].sum():,} ({df['is_illicit'].mean()*100:.1f}%)"
    )
    return df


def tokenize(batch, tokenizer, max_length: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits[:, 1]))
    preds = (probs >= 0.5).astype(int)

    auc_pr = average_precision_score(labels, probs)

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    idx = np.searchsorted(recall[::-1], 0.8)
    prec_at_r80 = float(precision[::-1][idx]) if idx < len(precision) else 0.0

    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    return {
        "auc_pr": round(auc_pr, 4),
        "prec_at_recall_80": round(prec_at_r80, 4),
        "f1_illicit": round(report.get("1", {}).get("f1-score", 0), 4),
        "precision_illicit": round(report.get("1", {}).get("precision", 0), 4),
        "recall_illicit": round(report.get("1", {}).get("recall", 0), 4),
        "accuracy": round(report.get("accuracy", 0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested Hydra config into MLflow-friendly scalar params."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, name))
        elif isinstance(value, str | int | float | bool) or value is None:
            flat[name] = value
    return flat


def resolve_max_samples(cfg: DictConfig) -> int | None:
    return cfg.training.get("max_samples", None) or cfg.data.get("max_samples", None)


def train(cfg: DictConfig):
    # ── Data ──────────────────────────────────────────────────────────────────
    max_samples = resolve_max_samples(cfg)
    df = load_data(cfg.data.csv_path, max_samples)

    train_df, eval_df = train_test_split(
        df,
        test_size=cfg.data.test_size,
        stratify=df["is_illicit"],
        random_state=cfg.data.seed,
    )
    print(f"Train: {len(train_df):,}  |  Eval: {len(eval_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    def to_hf_dataset(df_split):
        return Dataset.from_dict(
            {
                "text": df_split["memo_text"].tolist(),
                "label": df_split["is_illicit"].tolist(),
            }
        ).map(lambda b: tokenize(b, tokenizer, cfg.model.max_length), batched=True)

    train_ds = to_hf_dataset(train_df)
    eval_ds = to_hf_dataset(eval_df)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name,
        num_labels=cfg.model.num_labels,
    )

    # ── Class weights (handle 2% illicit imbalance) ───────────────────────────
    # We pass class_weight via a custom Trainer subclass below

    pos_weight = (len(train_df) - train_df["is_illicit"].sum()) / train_df[
        "is_illicit"
    ].sum()
    print(f"Positive class weight: {pos_weight:.1f}x")

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.train_batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metrics.primary_key,
        greater_is_better=True,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        fp16=cfg.training.fp16,
        report_to="none",
        seed=cfg.data.seed,
    )

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_experiment(cfg.training.mlflow_experiment)

    run_name = f"{cfg.experiment_name}_ep{cfg.training.epochs}"
    with mlflow.start_run(run_name=run_name):
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(flatten_config(resolved_cfg))
        mlflow.log_params(
            {
                "train_rows": len(train_df),
                "eval_rows": len(eval_df),
                "pos_weight": round(pos_weight, 2),
            }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # ── Evaluate & log ─────────────────────────────────────────────────────
        eval_results = trainer.evaluate()
        print("\nFinal eval results:")
        for k, v in eval_results.items():
            if not k.startswith("eval_runtime"):
                print(f"  {k}: {v}")

        mlflow.log_metrics(
            {
                k.replace("eval_", ""): v
                for k, v in eval_results.items()
                if isinstance(v, (int | float))
            }
        )

        # ── Save model ────────────────────────────────────────────────────────
        model_save_path = os.path.join(
            cfg.training.output_dir, cfg.training.model_subdir
        )
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"\nModel saved to: {model_save_path}")

        # ── Save metrics JSON (for DVC/CI eval gate) ──────────────────────────
        metrics = {
            k: v
            for k, v in eval_results.items()
            if isinstance(v, float) and not k.startswith("eval_runtime")
        }
        with open(cfg.training.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {cfg.training.metrics_output}")

        mlflow.log_artifact(cfg.training.metrics_output)
        mlflow.log_artifacts(model_save_path, artifact_path=cfg.training.model_subdir)

    return eval_results


# ─────────────────────────────────────────────────────────────────────────────
# HYDRA ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(
    config_path="../../conf", config_name="config_distilbert", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print("DistilBERT Hydra config:")
    print(OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
