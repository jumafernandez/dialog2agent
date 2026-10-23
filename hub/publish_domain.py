"""Publish a domain artifact to the Hugging Face Hub, with its card as README.md.

Usage:
    python hub/publish_domain.py DOMAIN_DIR REPO CARD [--message TEXT]

The training state of the adapter and the default card written by PEFT are not uploaded, since loading the agent
does not need them. The token is read from the local Hugging Face login.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

IGNORE = ["adapter/estado.pt", "adapter/README.md", "*.antes-*", "README.md"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", type=Path)
    ap.add_argument("repo")
    ap.add_argument("card", type=Path)
    ap.add_argument("--message", default="Release of the domain artifact", help="commit message")
    a = ap.parse_args()
    for required in ("domain.json", "structure/structure.npz", "memory/vectors.npy", "adapter/adapter_model.safetensors"):
        assert (a.domain / required).exists(), required
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=a.repo, folder_path=str(a.domain), ignore_patterns=IGNORE, commit_message=a.message)
    info = api.upload_file(path_or_fileobj=str(a.card), path_in_repo="README.md", repo_id=a.repo,
                           commit_message=a.message)
    print("PUBLISHED", a.repo, info, flush=True)


if __name__ == "__main__":
    main()
