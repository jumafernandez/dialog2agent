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
(calibration of the decision), ``max_tokens`` (prompt budget) and ``search_triggers`` (actions that call the search
tool).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_agent(path: str | Path, device: str | None = None, backend=None, base_model: str | None = None,
               turn_encoder: str | None = None):
    """Build an ``Agent`` from a domain artifact.

    ``base_model`` and ``turn_encoder`` replace the models named in ``domain.json``, for example with local copies
    of the same models in a cluster without access to the Hub.
    """
    from .agent import Agent
    from .backends import TransformersBackend
    from .decision import LLMScoringDecider
    from .memory import SituationMemory
    from .representations import Dialog2FlowEncoder, TraceDialogEncoder, TraceEncoder
    from .structure import InducedStructure
    from .tools import JsonDatabase, ToolRegistry

    path = Path(path)
    cfg = json.loads((path / "domain.json").read_text())
    if backend is None:
        adapter = path / "adapter" if (path / "adapter").exists() else None
        backend = TransformersBackend.from_pretrained(base_model or cfg["base_model"],
                                                      adapter=str(adapter) if adapter else None, device=device,
                                                      max_tokens=cfg.get("max_tokens", 2048))
    tools = ToolRegistry()
    if (path / "db").exists():
        tools.register(JsonDatabase(path / "db").as_tool(), triggered_by=cfg.get("search_triggers", []))
    structure = InducedStructure.load(path / "structure")
    return Agent(structure=structure, memory=SituationMemory.load(path / "memory"), backend=backend,
                 turn_encoder=Dialog2FlowEncoder(turn_encoder or cfg["turn_encoder"], device=device),
                 context_encoder=TraceEncoder(cfg.get("context_encoder"), device=device,
                                              revision=cfg.get("context_encoder_revision")),
                 dialogue_encoder=TraceDialogEncoder(cfg.get("dialogue_encoder"), device=device,
                                                     revision=cfg.get("dialogue_encoder_revision")),
                 decider=LLMScoringDecider(backend, structure, scale=cfg.get("decision_scale", 1.0)), tools=tools,
                 k=cfg.get("k", 10), conditioned=cfg.get("conditioned", True))
