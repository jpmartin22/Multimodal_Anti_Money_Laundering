"""
retrain_dag.py
==============
Airflow DAG: AML multimodal weekly retraining pipeline.

Steps
-----
1. data_pull    — dvc pull + preprocessing
2. train_fusion — retrain late-fusion MLP
3. eval_gate    — AUC-PR gate (>= 0.80); branches to promote or fail
4. promote_model — upload to GCS registry, mark as latest

Local test
----------
    pip install apache-airflow
    airflow db migrate
    airflow dags test retrain_dag 2024-01-01
    airflow tasks test retrain_dag train_fusion 2024-01-01

Env vars
--------
    GCS_MODEL_BUCKET   GCS bucket (default: aml-model-registry)
    MODEL_VERSION      override version tag (default: YYYY-MM-DD)
    AML_THRESHOLD      AUC-PR gate threshold (default: 0.80)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

_DEFAULT_ARGS = {
    "owner": "rajani",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

_GCS_BUCKET = os.getenv("GCS_MODEL_BUCKET", "aml-model-registry")
_AUC_PR_THRESHOLD = float(os.getenv("AML_THRESHOLD", "0.80"))


# ---------------------------------------------------------------------------
# Callables
# ---------------------------------------------------------------------------


def _check_gate(**context) -> str:
    """Branch on AUC-PR gate. Returns next task_id."""
    import json
    from pathlib import Path

    path = Path("reports/metrics.json")
    if not path.exists():
        raise FileNotFoundError(f"Metrics file missing: {path}")

    auc_pr = json.loads(path.read_text()).get("auc_pr", 0.0)
    context["ti"].xcom_push(key="auc_pr", value=auc_pr)

    if auc_pr >= _AUC_PR_THRESHOLD:
        context["ti"].log.info("Gate PASSED: auc_pr=%.4f", auc_pr)
        return "promote_model"

    context["ti"].log.error(
        "Gate FAILED: auc_pr=%.4f < threshold=%.2f", auc_pr, _AUC_PR_THRESHOLD
    )
    return "gate_failed"


def _fail_gate(**context) -> None:
    auc_pr = context["ti"].xcom_pull(task_ids="eval_gate", key="auc_pr")
    raise ValueError(
        f"AUC-PR gate failed: {auc_pr:.4f} < {_AUC_PR_THRESHOLD}. Model not promoted."
    )


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="retrain_dag",
    description="Weekly AML retraining: data pull → train → eval gate → promote",
    default_args=_DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["aml", "retraining", "phase3"],
) as dag:
    # 1. Data pull
    data_pull = BashOperator(
        task_id="data_pull",
        bash_command=(
            "set -euo pipefail\n"
            "echo '=== data_pull: pulling DVC data ==='\n"
            "dvc pull --run-cache\n"
            "python -m multimodal_anti_money_laundering.data.make_dataset\n"
            "echo '=== data_pull: done ==='"
        ),
    )

    # 2. Train fusion model
    train_fusion = BashOperator(
        task_id="train_fusion",
        bash_command=(
            "set -euo pipefail\n"
            "echo '=== train_fusion: retraining fusion MLP ==='\n"
            "python -m multimodal_anti_money_laundering.train_fusion\n"
            "echo '=== train_fusion: done — metrics at reports/metrics.json ==='"
        ),
    )

    # 3. Eval gate (branches)
    eval_gate = BranchPythonOperator(
        task_id="eval_gate",
        python_callable=_check_gate,
    )

    gate_failed = PythonOperator(
        task_id="gate_failed",
        python_callable=_fail_gate,
    )

    # 4. Promote to GCS
    promote_model = BashOperator(
        task_id="promote_model",
        bash_command=(
            "set -euo pipefail\n"
            'VERSION="${MODEL_VERSION:-$(date +%Y-%m-%d)}"\n'
            'echo "=== promote_model: uploading fusion v${VERSION} to GCS ==="\n'
            "python scripts/gcs_model_registry.py upload \\\n"
            "    --model fusion \\\n"
            f"    --bucket {_GCS_BUCKET} \\\n"
            '    --version "$VERSION" \\\n'
            "    --local-dir models/fusion/\n"
            "python scripts/gcs_model_registry.py promote \\\n"
            "    --model fusion \\\n"
            f"    --bucket {_GCS_BUCKET} \\\n"
            '    --version "$VERSION"\n'
            'echo "=== promote_model: v${VERSION} is now latest ==="'
        ),
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # Dependencies
    data_pull >> train_fusion >> eval_gate >> [promote_model, gate_failed]
