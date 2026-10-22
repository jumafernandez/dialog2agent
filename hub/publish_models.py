"""Publish converted models to the Hugging Face Hub (one public repository per model).

Usage:
    python hub/publish_models.py EXPORT_DIR NAMESPACE MODEL [MODEL ...] [--message TEXT]

Each ``EXPORT_DIR/MODEL`` folder (configuration, weights, code and README.md card) is uploaded to
``NAMESPACE/MODEL``. The token is read from the local Hugging Face login.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", type=Path)
    ap.add_argument("namespace")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--message", default="Release in transformers format", help="commit message")
    a = ap.parse_args()
    api = HfApi()
    for name in a.models:
        folder = a.export / name
        assert (folder / "README.md").exists() and (folder / "model.safetensors").exists(), folder
        repo = f"{a.namespace}/{name}"
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        info = api.upload_folder(repo_id=repo, folder_path=str(folder), commit_message=a.message)
        print("PUBLISHED", repo, info.oid if hasattr(info, "oid") else info, flush=True)


if __name__ == "__main__":
    main()
