"""
push_to_spaces.py
=================
Member D (Rajani) — Phase 3 §3 HuggingFace Spaces Deployment

Pushes the AML serving API to a HuggingFace Space (Docker SDK).

Prerequisites
-------------
    pip install huggingface_hub
    huggingface-cli login          # or set HF_TOKEN env var

Usage
-----
    # First time — create the Space and upload:
    python deploy/push_to_spaces.py --username <your-hf-username>

    # Update an existing Space after code changes:
    python deploy/push_to_spaces.py --username <your-hf-username> --update

    # Custom Space name:
    python deploy/push_to_spaces.py --username <your-hf-username> --space-name my-aml-api
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_space_directory(tmp_dir: Path) -> None:
    """Copy all files needed by the HF Space into tmp_dir."""

    # 1. HF Space README (contains YAML frontmatter — required by HF)
    shutil.copy(PROJECT_ROOT / "deploy" / "huggingface" / "README.md",
                tmp_dir / "README.md")

    # 2. Dockerfile (HF port 7860 variant)
    shutil.copy(PROJECT_ROOT / "dockerfiles" / "Dockerfile.hf",
                tmp_dir / "Dockerfile")

    # 3. Python package source
    shutil.copytree(PROJECT_ROOT / "src", tmp_dir / "src")

    # 4. Serving requirements
    shutil.copy(PROJECT_ROOT / "requirements.serve.txt",
                tmp_dir / "requirements.serve.txt")

    # 5. pyproject.toml (needed for pip install -e .)
    shutil.copy(PROJECT_ROOT / "pyproject.toml",
                tmp_dir / "pyproject.toml")

    # 6. .env.example (safe — no secrets)
    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        shutil.copy(env_example, tmp_dir / ".env.example")

    print(f"Space directory built at: {tmp_dir}")
    print("Files staged:")
    for f in sorted(tmp_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(tmp_dir)}")


def push_to_spaces(username: str, space_name: str, update: bool) -> str:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    repo_id = f"{username}/{space_name}"
    api = HfApi()

    if not update:
        print(f"Creating Space: {repo_id} ...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        print(f"Space created: https://huggingface.co/spaces/{repo_id}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        build_space_directory(tmp_path)

        print(f"\nUploading to {repo_id} ...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="space",
            commit_message="deploy: AML serving API via push_to_spaces.py",
        )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\nDeploy complete.")
    print(f"Space URL : {url}")
    print(f"API docs  : {url}/docs")
    print(f"Health    : {url}/health")
    print("\nNote: HF Spaces builds the Docker image automatically (2-5 min).")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Push AML API to HuggingFace Spaces")
    parser.add_argument("--username",   required=True, help="HuggingFace username")
    parser.add_argument("--space-name", default="aml-multimodal-scorer",
                        help="Space name (default: aml-multimodal-scorer)")
    parser.add_argument("--update", action="store_true",
                        help="Skip Space creation, just upload files")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Tip: set HF_TOKEN env var to avoid interactive login prompt.")
        print("     export HF_TOKEN=hf_your_token_here\n")

    push_to_spaces(
        username=args.username,
        space_name=args.space_name,
        update=args.update,
    )


if __name__ == "__main__":
    main()
