"""Tools the agent can call to act on its environment, such as searching a database or making a booking.

A tool is triggered by the decided action (for example, actions that inform or offer an entity), and its arguments
are the constraints extracted by the language model from the dialogue. Its result enters the prompt of the writer, so
that the entities the agent mentions come from the environment and not from other conversations.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import ToolCall


@dataclass
class Tool:
    """A callable with a name, a description and a JSON schema of its arguments."""

    name: str
    description: str
    parameters: dict
    function: Callable[..., Any]

    def __call__(self, **arguments) -> ToolCall:
        return ToolCall(name=self.name, arguments=arguments, result=self.function(**arguments))


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    triggers: dict[str, str] = field(default_factory=dict)  # action-name keyword -> tool name

    def register(self, tool: Tool, triggered_by: list[str] | None = None) -> None:
        self.tools[tool.name] = tool
        for keyword in triggered_by or []:
            self.triggers[keyword] = tool.name

    def for_action(self, action_name: str) -> Tool | None:
        """The tool triggered by an action, if any."""
        for keyword, name in self.triggers.items():
            if keyword in action_name:
                return self.tools[name]
        return None


class JsonDatabase:
    """A database of entities per domain stored as JSON files (``<domain>_db.json``), as in MultiWOZ and SpokenWOZ."""

    def __init__(self, path: str | Path):
        self.tables = {p.name.removesuffix("_db.json"): json.loads(p.read_text())
                       for p in Path(path).glob("*_db.json")}

    def search(self, domain: str, constraints: dict[str, str], limit: int = 5) -> list[dict]:
        rows = self.tables.get(domain, [])
        match = [r for r in rows if all(str(r.get(k, "")).lower() == str(v).lower() for k, v in constraints.items())]
        return match[:limit]

    def as_tool(self) -> Tool:
        return Tool(name="search", description="Search entities of a domain that satisfy the user constraints.",
                    parameters={"type": "object", "properties": {"domain": {"type": "string"},
                                                                 "constraints": {"type": "object"}},
                                "required": ["domain", "constraints"]},
                    function=lambda domain, constraints: self.search(domain, constraints))
