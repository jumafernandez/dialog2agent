"""Deployment of an agent on Modal, with a serverless GPU that scales to zero when the agent is not used.

Usage, from the root of the repository and with the Modal client authenticated:

    modal volume create dialog2agent-domains
    modal volume put dialog2agent-domains DOMAIN_DIR /spokenwoz     # the domain artifact, once
    modal run deploy/modal_app.py::download                          # base model and encoders, once
    modal serve deploy/modal_app.py                                  # temporary URL while the command runs
    modal deploy deploy/modal_app.py                                 # permanent URL

Everything that determines the behavior of the agent is fixed: the versions of the libraries in the image, the
commits of the models on the Hugging Face Hub and the domain artifact in the volume.
"""
from pathlib import Path

import modal

APP = "dialog2agent-spokenwoz"
DOMAIN = "spokenwoz"
GPU = "L4"
BASE_MODEL, BASE_REVISION = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TURN_ENCODER = "sergioburdisso/dialog2flow-joint-bert-base"
MODELS, DOMAINS = "/models", "/domains"
SOURCE = Path(__file__).resolve().parent.parent / "src" / "dialog2agent"

# The same versions with which the agent was evaluated.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==5.14.1", "peft==0.21.2", "sentence-transformers==5.6.0",
                 "accelerate==1.15.0", "huggingface_hub==1.24.0", "numpy==2.4.3", "fastapi[standard]==0.159.1")
    .env({"HF_HOME": f"{MODELS}/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_dir(SOURCE, "/root/dialog2agent")
)
models = modal.Volume.from_name("dialog2agent-models", create_if_missing=True)
domains = modal.Volume.from_name("dialog2agent-domains", create_if_missing=True)
app = modal.App(APP)


def local_dir(repo: str) -> str:
    return f"{MODELS}/{repo.split('/')[-1]}"


@app.function(image=image, volumes={MODELS: models, DOMAINS: domains}, timeout=3600)
def download() -> None:
    """Download the base model, the turn encoder and the TRACE encoders at the commits of the domain artifact."""
    import json

    from huggingface_hub import snapshot_download

    from dialog2agent.representations import REVISIONS
    snapshot_download(BASE_MODEL, revision=BASE_REVISION, local_dir=local_dir(BASE_MODEL))
    snapshot_download(TURN_ENCODER, local_dir=local_dir(TURN_ENCODER))
    cfg = json.loads(Path(f"{DOMAINS}/{DOMAIN}/domain.json").read_text())
    pinned = {(cfg["context_encoder"], cfg["context_encoder_revision"]),
              (cfg["dialogue_encoder"], cfg["dialogue_encoder_revision"]), *REVISIONS.items()}
    for repo, revision in sorted(pinned):
        snapshot_download(repo, revision=revision)
    models.commit()
    print("DOWNLOAD-OK", sorted(pinned), flush=True)


@app.cls(image=image, gpu=GPU, volumes={MODELS: models, DOMAINS: domains}, scaledown_window=300, timeout=600,
         max_containers=1)
@modal.concurrent(max_inputs=8)
class Service:
    """One replica at most, so that the cost is bounded; requests wait for the agent in turn."""

    @modal.enter()
    def load(self) -> None:
        import os
        import time
        os.environ["HF_HUB_OFFLINE"] = "1"
        from dialog2agent import Agent
        start = time.time()
        self.agent = Agent.from_pretrained(f"{DOMAINS}/{DOMAIN}", device="cuda", base_model=local_dir(BASE_MODEL),
                                           turn_encoder=local_dir(TURN_ENCODER))
        print(f"agent loaded in {time.time() - start:.0f} s", flush=True)

    @modal.asgi_app()
    def web(self):
        from dialog2agent.server import create_app
        return create_app(self.agent)
