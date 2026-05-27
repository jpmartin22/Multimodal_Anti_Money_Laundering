"""
gcs_model_registry.py
======================
Push trained model artifacts to a GCS bucket and retrieve them.

Implements a lightweight versioning scheme:
  gs://<bucket>/models/<model_name>/<version>/

Usage
-----
    # Upload the fusion model after training
    python scripts/gcs_model_registry.py upload \
        --model fusion \
        --version 1.0.0 \
        --local-dir models/fusion/

    # Download a specific version
    python scripts/gcs_model_registry.py download \
        --model fusion \
        --version 1.0.0 \
        --dest models/fusion/

    # List available versions
    python scripts/gcs_model_registry.py list --model fusion

    # Promote a version as 'latest' alias
    python scripts/gcs_model_registry.py promote \
        --model fusion \
        --version 1.0.0

Prerequisites
-------------
    pip install google-cloud-storage
    gcloud auth application-default login   # or set GOOGLE_APPLICATION_CREDENTIALS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

GCS_BUCKET = os.getenv("GCS_MODEL_BUCKET", "aml-model-registry")
GCS_PREFIX = "models"


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def _get_client():
    try:
        from google.cloud import storage
    except ImportError:
        log.error(
            "google-cloud-storage not installed. Run: pip install google-cloud-storage"
        )
        sys.exit(1)
    return storage.Client()


def _model_prefix(model_name: str, version: str) -> str:
    return f"{GCS_PREFIX}/{model_name}/{version}/"


def _latest_alias_blob(model_name: str) -> str:
    return f"{GCS_PREFIX}/{model_name}/latest.json"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_upload(args: argparse.Namespace) -> None:
    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        log.error("Local directory does not exist: %s", local_dir)
        sys.exit(1)

    files = [f for f in local_dir.rglob("*") if f.is_file()]
    if not files:
        log.error("No files found in %s", local_dir)
        sys.exit(1)

    client = _get_client()
    bucket = client.bucket(args.bucket)
    prefix = _model_prefix(args.model, args.version)

    log.info("Uploading %d file(s) to gs://%s/%s", len(files), args.bucket, prefix)
    for local_path in sorted(files):
        rel = local_path.relative_to(local_dir)
        blob_name = f"{prefix}{rel}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        log.info("  uploaded: %s → gs://%s/%s", rel, args.bucket, blob_name)

    # Write a metadata manifest
    manifest = {
        "model": args.model,
        "version": args.version,
        "uploaded_at": datetime.now(UTC).isoformat(),
        "files": [str(f.relative_to(local_dir)) for f in sorted(files)],
        "uploader": os.getenv("USER", "unknown"),
    }
    manifest_blob = bucket.blob(f"{prefix}manifest.json")
    manifest_blob.upload_from_string(
        json.dumps(manifest, indent=2), content_type="application/json"
    )
    log.info("Manifest written: gs://%s/%smanifest.json", args.bucket, prefix)
    log.info("Upload complete.")


def cmd_download(args: argparse.Namespace) -> None:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    bucket = client.bucket(args.bucket)

    # Resolve 'latest' alias
    version = args.version
    if version == "latest":
        alias_blob = bucket.blob(_latest_alias_blob(args.model))
        if not alias_blob.exists():
            log.error(
                "No 'latest' alias for model '%s'. Run 'promote' first.", args.model
            )
            sys.exit(1)
        alias_data = json.loads(alias_blob.download_as_text())
        version = alias_data["version"]
        log.info("Resolved 'latest' → version %s", version)

    prefix = _model_prefix(args.model, version)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        log.error("No artifacts found at gs://%s/%s", args.bucket, prefix)
        sys.exit(1)

    log.info("Downloading %d file(s) from gs://%s/%s", len(blobs), args.bucket, prefix)
    for blob in blobs:
        rel = blob.name[len(prefix) :]
        local_path = dest / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        log.info("  downloaded: %s → %s", blob.name, local_path)

    log.info("Download complete → %s", dest)


def cmd_list(args: argparse.Namespace) -> None:
    client = _get_client()
    bucket = client.bucket(args.bucket)
    prefix = f"{GCS_PREFIX}/{args.model}/"

    # Collect unique version folders
    versions: dict[str, str] = {}
    for blob in bucket.list_blobs(prefix=prefix):
        parts = blob.name[len(prefix) :].split("/")
        if len(parts) >= 1 and parts[0]:
            version_dir = parts[0]
            if version_dir != "latest.json" and version_dir not in versions:
                versions[version_dir] = (
                    blob.time_created.isoformat() if blob.time_created else ""
                )

    if not versions:
        log.info("No versions found for model '%s' in gs://%s", args.model, args.bucket)
        return

    log.info("Versions of '%s' in gs://%s:", args.model, args.bucket)
    for ver in sorted(versions):
        log.info("  %s  (first blob: %s)", ver, versions[ver])

    # Check for latest alias
    alias_blob = bucket.blob(_latest_alias_blob(args.model))
    if alias_blob.exists():
        alias = json.loads(alias_blob.download_as_text())
        log.info(
            "latest → %s  (promoted: %s)",
            alias["version"],
            alias.get("promoted_at", "?"),
        )


def cmd_promote(args: argparse.Namespace) -> None:
    client = _get_client()
    bucket = client.bucket(args.bucket)
    prefix = _model_prefix(args.model, args.version)

    # Verify the version exists
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        log.error(
            "Version '%s' of model '%s' not found in gs://%s",
            args.version,
            args.model,
            args.bucket,
        )
        sys.exit(1)

    alias = {
        "model": args.model,
        "version": args.version,
        "promoted_at": datetime.now(UTC).isoformat(),
        "promoted_by": os.getenv("USER", "unknown"),
    }
    alias_blob = bucket.blob(_latest_alias_blob(args.model))
    alias_blob.upload_from_string(
        json.dumps(alias, indent=2), content_type="application/json"
    )
    log.info(
        "Promoted '%s' v%s as 'latest' in gs://%s",
        args.model,
        args.version,
        args.bucket,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GCS model registry — upload, download, list, promote"
    )
    parser.add_argument(
        "--bucket",
        default=GCS_BUCKET,
        help=f"GCS bucket name (default: {GCS_BUCKET}, override with GCS_MODEL_BUCKET env var)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    up = sub.add_parser("upload", help="Upload model artifacts to GCS")
    up.add_argument(
        "--model", required=True, help="Model name (e.g. fusion, distilbert)"
    )
    up.add_argument("--version", required=True, help="Semantic version (e.g. 1.0.0)")
    up.add_argument(
        "--local-dir", required=True, help="Local directory with model files"
    )

    # download
    dl = sub.add_parser("download", help="Download model artifacts from GCS")
    dl.add_argument("--model", required=True)
    dl.add_argument("--version", default="latest", help="Version tag or 'latest'")
    dl.add_argument("--dest", required=True, help="Local destination directory")

    # list
    ls = sub.add_parser("list", help="List available versions for a model")
    ls.add_argument("--model", required=True)

    # promote
    pr = sub.add_parser("promote", help="Mark a version as 'latest'")
    pr.add_argument("--model", required=True)
    pr.add_argument("--version", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "upload": cmd_upload,
        "download": cmd_download,
        "list": cmd_list,
        "promote": cmd_promote,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
