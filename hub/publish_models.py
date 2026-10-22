"""Publish converted models to the Hugging Face Hub (one public repository per model).

Usage:
    python hub/publish_models.py EXPORT_DIR NAMESPACE MODEL [MODEL ...]

Each ``EXPORT_DIR/MODEL`` folder (configuration, weights, code and README.md card) is uploaded to
``NAMESPACE/MODEL``. The token is read from the local Hugging Face login.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    export, namespace, models = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    api = HfApi()
    for name in models:
        folder = export / name
        assert (folder / "README.md").exists() and (folder / "model.safetensors").exists(), folder
        repo = f"{namespace}/{name}"
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        info = api.upload_folder(repo_id=repo, folder_path=str(folder), commit_message="Release in transformers format")
        print("PUBLISHED", repo, info.oid if hasattr(info, "oid") else info, flush=True)


if __name__ == "__main__":
    main()
