"""
vertex_ai_training.py
=====================
Submit a custom training job to Vertex AI using the project's
Docker image stored in Artifact Registry.

The job runs the late-fusion training pipeline inside the container
and uploads the resulting model artifacts to the GCS model registry.

Usage
-----
    # Submit fusion MLP training (default)
    python scripts/vertex_ai_training.py submit

    # Submit with custom config
    python scripts/vertex_ai_training.py submit \
        --model fusion \
        --machine-type n1-standard-8 \
        --accelerator-type NVIDIA_TESLA_T4 \
        --accelerator-count 1 \
        --version 1.1.0

    # Check job status
    python scripts/vertex_ai_training.py status --job-id <job_resource_name>

    # List recent jobs
    python scripts/vertex_ai_training.py list

Prerequisites
-------------
    pip install google-cloud-aiplatform
    gcloud auth application-default login   # or GOOGLE_APPLICATION_CREDENTIALS

Environment variables (or set via --flag):
    GCP_PROJECT_ID    GCP project ID
    GCP_REGION        Vertex AI region  (default: us-central1)
    AR_REPO           Artifact Registry repo name  (default: aml-images)
    GCS_MODEL_BUCKET  Bucket for model upload after training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via env vars or CLI flags)
# ---------------------------------------------------------------------------

GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
AR_REPO = os.getenv("AR_REPO", "aml-images")
GCS_BUCKET = os.getenv("GCS_MODEL_BUCKET", "aml-model-registry")

# The training image pushed by the docker-build.yml workflow
_IMAGE_TMPL = "{region}-docker.pkg.dev/{project}/{repo}/aml-api:latest"

_MODEL_TRAIN_COMMANDS = {
    "fusion": [
        "python",
        "-m",
        "multimodal_anti_money_laundering.train_fusion",
        "--output-dir",
        "/gcs/$(GCS_MODEL_BUCKET)/models/fusion/$(VERSION)/",
    ],
    "distilbert": [
        "python",
        "-m",
        "multimodal_anti_money_laundering.train_distilbert",
    ],
    "graphsage": [
        "python",
        "-m",
        "multimodal_anti_money_laundering.train_graphsage",
    ],
    "bilstm": [
        "python",
        "-m",
        "multimodal_anti_money_laundering.train_bilstm",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_aiplatform():
    try:
        from google.cloud import aiplatform

        return aiplatform
    except ImportError:
        log.error("google-cloud-aiplatform not installed.")
        log.error("Run: pip install google-cloud-aiplatform")
        sys.exit(1)


def _resolve_image(project: str, region: str, repo: str) -> str:
    return _IMAGE_TMPL.format(region=region, project=project, repo=repo)


def _version_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_submit(args: argparse.Namespace) -> None:
    aip = _require_aiplatform()

    project = args.project or GCP_PROJECT
    region = args.region
    if not project:
        log.error("GCP project ID required. Set --project or GCP_PROJECT_ID env var.")
        sys.exit(1)

    version = args.version or _version_tag()
    image_uri = _resolve_image(project, region, args.ar_repo)
    display_name = f"aml-{args.model}-{version}"

    log.info("Initialising Vertex AI SDK (project=%s, region=%s)", project, region)
    aip.init(project=project, location=region)

    # Worker pool spec
    worker_pool = {
        "machine_spec": {
            "machine_type": args.machine_type,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": image_uri,
            "command": _MODEL_TRAIN_COMMANDS.get(
                args.model, _MODEL_TRAIN_COMMANDS["fusion"]
            ),
            "env": [
                {"name": "GCS_MODEL_BUCKET", "value": args.bucket},
                {"name": "VERSION", "value": version},
                {"name": "MLFLOW_TRACKING_URI", "value": f"gs://{args.bucket}/mlruns"},
                {"name": "AML_SEED", "value": "42"},
            ],
        },
    }

    if args.accelerator_type and args.accelerator_type.upper() != "NONE":
        worker_pool["machine_spec"]["accelerator_type"] = args.accelerator_type
        worker_pool["machine_spec"]["accelerator_count"] = args.accelerator_count

    log.info("Submitting custom training job: %s", display_name)
    log.info("  image     : %s", image_uri)
    log.info("  machine   : %s", args.machine_type)
    log.info("  model     : %s  version=%s", args.model, version)

    job = aip.CustomJob(
        display_name=display_name,
        worker_pool_specs=[worker_pool],
    )

    if args.async_mode:
        job.submit()
        log.info("Job submitted asynchronously.")
        log.info("  Resource name : %s", job.resource_name)
        log.info(
            "  Console URL   : https://console.cloud.google.com/vertex-ai/training/custom-jobs"
        )
        _save_job_ref(job.resource_name, display_name, version, args.model)
    else:
        log.info("Waiting for job to complete (this can take 30–120 min) ...")
        job.run(sync=True)
        log.info("Job finished with state: %s", job.state)
        _save_job_ref(job.resource_name, display_name, version, args.model)
        if str(job.state) != "JobState.JOB_STATE_SUCCEEDED":
            sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    aip = _require_aiplatform()

    project = args.project or GCP_PROJECT
    if not project:
        log.error("GCP project ID required.")
        sys.exit(1)

    aip.init(project=project, location=args.region)
    job = aip.CustomJob.get(resource_name=args.job_id)
    log.info("Job: %s", job.display_name)
    log.info("State: %s", job.state)
    if hasattr(job, "error") and job.error:
        log.error("Error: %s", job.error)


def cmd_list(args: argparse.Namespace) -> None:
    aip = _require_aiplatform()

    project = args.project or GCP_PROJECT
    if not project:
        log.error("GCP project ID required.")
        sys.exit(1)

    aip.init(project=project, location=args.region)
    jobs = aip.CustomJob.list(
        filter='display_name="aml-*"', order_by="create_time desc"
    )
    if not jobs:
        log.info("No AML training jobs found.")
        return

    log.info("Recent AML training jobs (newest first):")
    for j in jobs[: args.limit]:
        log.info("  [%s] %s  —  %s", j.state, j.display_name, j.resource_name)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _save_job_ref(
    resource_name: str, display_name: str, version: str, model: str
) -> None:
    ref_path = "reports/vertex_ai_jobs.jsonl"
    entry = {
        "resource_name": resource_name,
        "display_name": display_name,
        "model": model,
        "version": version,
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    with open(ref_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    log.info("Job reference saved to %s", ref_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit and manage Vertex AI training jobs"
    )
    parser.add_argument(
        "--project", default="", help="GCP project ID (or set GCP_PROJECT_ID)"
    )
    parser.add_argument("--region", default=GCP_REGION)
    parser.add_argument(
        "--ar-repo", default=AR_REPO, help="Artifact Registry repo name"
    )
    parser.add_argument(
        "--bucket", default=GCS_BUCKET, help="GCS bucket for model output"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # submit
    s = sub.add_parser("submit", help="Submit a training job")
    s.add_argument(
        "--model",
        default="fusion",
        choices=list(_MODEL_TRAIN_COMMANDS.keys()),
        help="Which model to train",
    )
    s.add_argument("--version", default="", help="Version tag (default: timestamp)")
    s.add_argument("--machine-type", default="n1-standard-4")
    s.add_argument(
        "--accelerator-type",
        default="NONE",
        help="e.g. NVIDIA_TESLA_T4, NVIDIA_TESLA_V100, NONE",
    )
    s.add_argument("--accelerator-count", type=int, default=1)
    s.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Submit without waiting for completion",
    )

    # status
    st = sub.add_parser("status", help="Check job status")
    st.add_argument("--job-id", required=True, help="Vertex AI job resource name")

    # list
    ls = sub.add_parser("list", help="List recent training jobs")
    ls.add_argument("--limit", type=int, default=10)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    commands = {"submit": cmd_submit, "status": cmd_status, "list": cmd_list}
    commands[args.command](args)


if __name__ == "__main__":
    main()
