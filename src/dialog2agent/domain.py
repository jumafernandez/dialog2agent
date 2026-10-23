"""Domain artifacts: everything the agent learns from a collection of dialogues, in one directory.

```
my-domain/
  domain.json        # base model, adapter, encoders and agent settings
  structure/         # induced structure (structure.npz, structure.json)
  memory/            # situations (vectors.npy, entries.jsonl, memory.json)
  adapter/           # LoRA adapter of the language model (optional)
  db/                # database of the tools (optional, <domain>_db.json)
```

The same agent code loads any artifact, so a new domain only requires building its artifact from its own dialogues,
without annotations.

Keys of ``domain.json``: ``base_model``, ``turn_encoder``, ``context_encoder`` and ``dialogue_encoder`` (model
names, the last two with optional ``*_revision`` commits), ``k`` and ``conditioned`` (retrieval), ``decision_scale``
(calibration of the decision), ``max_tokens`` (prompt budget), ``search_triggers`` (actions that call the search
tool), ``scope`` (thresholds of the scope of the domain, see ``dialog2agent.scope``) and ``data`` (name, citation, URL
and license of the dialogues from which the artifact was built, which the web service shows as attribution).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_agent(path: str | Path, device: str | None = None, backend=None, base_model: str | None = None,
               turn_encoder: str | None = None, adapter: bool = True, decision: str = "score",
               revision: str | None = None):
    """Build an ``Agent`` from a domain artifact, given as a local directory or as a repository of the Hugging Face
    Hub, which is downloaded at ``revision``.

    ``base_model`` and ``turn_encoder`` replace the models named in ``domain.json``, for example with local copies
    of the same models in a cluster without access to the Hub. ``adapter=False`` loads the base model without the
    adapter of the artifact, and ``decision`` selects how the action is decided (see ``Agent``).
    """
    from .agent import Agent
    from .backends import TransformersBackend
    from .decision import LLMScoringDecider
    from .memory import SituationMemory
    from .representations import Dialog2FlowEncoder, TraceDialogEncoder, TraceEncoder
    from .scope import Scope
    from .structure import InducedStructure
    from .tools import JsonDatabase, ToolRegistry

    path = Path(path)
    if not path.exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(path), revision=revision))
    cfg = json.loads((path / "domain.json").read_text())
    with_adapter = None  # unknown when the backend is given
    if backend is None:
        lora = path / "adapter" if adapter and (path / "adapter").exists() else None
        with_adapter = lora is not None
        backend = TransformersBackend.from_pretrained(base_model or cfg["base_model"],
                                                      adapter=str(lora) if lora else None, device=device,
                                                      max_tokens=cfg.get("max_tokens", 2048))
    tools = ToolRegistry()
    if (path / "db").exists():
        tools.register(JsonDatabase(path / "db").as_tool(), triggered_by=cfg.get("search_triggers", []))
    structure = InducedStructure.load(path / "structure")
    agent = Agent(structure=structure, memory=SituationMemory.load(path / "memory"), backend=backend,
                  turn_encoder=Dialog2FlowEncoder(turn_encoder or cfg["turn_encoder"], device=device),
                  context_encoder=TraceEncoder(cfg.get("context_encoder"), device=device,
                                               revision=cfg.get("context_encoder_revision")),
                  dialogue_encoder=TraceDialogEncoder(cfg.get("dialogue_encoder"), device=device,
                                                      revision=cfg.get("dialogue_encoder_revision")),
                  decider=LLMScoringDecider(backend, structure, scale=cfg.get("decision_scale", 1.0)), tools=tools,
                  k=cfg.get("k", 10), conditioned=cfg.get("conditioned", True), decision=decision,
                  scope=Scope(**cfg["scope"]) if cfg.get("scope") else None)
    agent.metadata = {**cfg, "adapter": with_adapter, "decision": decision}
    return agent
