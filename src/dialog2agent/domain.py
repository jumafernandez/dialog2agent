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
"""
from __future__ import annotations

import json
from pathlib import Path


def load_agent(path: str | Path, device: str | None = None, backend=None):
    """Build an ``Agent`` from a domain artifact."""
    from .agent import Agent
    from .backends import TransformersBackend
    from .memory import SituationMemory
    from .representations import Dialog2FlowEncoder, TraceDialogEncoder, TraceEncoder
    from .structure import InducedStructure
    from .tools import JsonDatabase, ToolRegistry

    path = Path(path)
    cfg = json.loads((path / "domain.json").read_text())
    if backend is None:
        adapter = path / "adapter" if (path / "adapter").exists() else None
        backend = TransformersBackend.from_pretrained(cfg["base_model"], adapter=str(adapter) if adapter else None,
                                                      device=device)
    tools = ToolRegistry()
    if (path / "db").exists():
        tools.register(JsonDatabase(path / "db").as_tool(), triggered_by=cfg.get("search_triggers", []))
    return Agent(structure=InducedStructure.load(path / "structure"), memory=SituationMemory.load(path / "memory"),
                 backend=backend, turn_encoder=Dialog2FlowEncoder(cfg["turn_encoder"], device=device),
                 context_encoder=TraceEncoder(cfg["context_encoder"], device=device),
                 dialogue_encoder=TraceDialogEncoder(cfg["dialogue_encoder"], device=device), tools=tools,
                 k=cfg.get("k", 10), conditioned=cfg.get("conditioned", True))
