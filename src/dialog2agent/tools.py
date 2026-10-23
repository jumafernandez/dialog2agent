"""Tools the agent can call to act on its environment, such as searching a database or making a booking.

A tool is triggered by the decided action: the domain artifact lists the induced actions that call each tool, for
example the actions in which the operators named an entity of the database or confirmed a booking. Its arguments are
the constraints of the user, extracted by the language model from the dialogue (see ``dialog2agent.extraction``), and
its result enters the prompt of the writer, so that the entities the agent mentions come from the environment and not
from other conversations.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Action, ToolCall

# fields that identify or describe an entity, which a user does not use as constraints
DESCRIPTIVE = {"id", "location", "introduction", "signature", "phone", "postcode", "address", "price", "duration",
               "trainID", "entrance fee", "openhours", "takesbookings"}
MAX_VALUES = 150  # fields with at most this many distinct values are categorical
EMPTY = {"", "dontcare", "dont care", "any", "none", "not mentioned"}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9: ]", " ", str(value).lower().replace("&", " and "))).strip()


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
    actions: dict[int, str] = field(default_factory=dict)  # induced action id -> tool name

    def register(self, tool: Tool, triggered_by: list[str] | None = None, actions: list[int] | None = None) -> None:
        self.tools[tool.name] = tool
        for keyword in triggered_by or []:
            self.triggers[keyword] = tool.name
        for action in actions or []:
            self.actions[int(action)] = tool.name

    def for_action(self, action: Action | str) -> Tool | None:
        """The tool triggered by an action, by its id when the artifact lists it or else by keywords of its name."""
        if isinstance(action, Action) and action.id in self.actions:
            return self.tools[self.actions[action.id]]
        name = action.name if isinstance(action, Action) else action
        for keyword, tool in self.triggers.items():
            if keyword in name:
                return self.tools[tool]
        return None


class JsonDatabase:
    """A database of entities per domain stored as JSON files (``<domain>_db.json``), as in MultiWOZ and SpokenWOZ.

    A file that holds a single object of value lists, such as the taxis of MultiWOZ, describes a service without
    entities, whose bookings are assembled from those lists. An optional ``booking.json`` declares the parameters of
    the booking of each domain (for example ``{"restaurant": ["people", "day", "time"]}``), which belong to the
    booking interface and not to the database.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        self.tables: dict[str, list[dict]] = {}
        self.services: dict[str, dict[str, list]] = {}
        for p in sorted(path.glob("*_db.json")):
            rows = json.loads(p.read_text())
            domain = p.name.removesuffix("_db.json")
            if rows and isinstance(rows, list) and all(isinstance(v, list) for v in rows[0].values()):
                self.services[domain] = rows[0]
            elif isinstance(rows, list):
                self.tables[domain] = rows
        booking = path / "booking.json"
        self.booking: dict[str, list[str]] = json.loads(booking.read_text()) if booking.exists() else {}

    def schema(self) -> dict[str, dict[str, list[str] | None]]:
        """The slots of each domain that a user can constrain, with their values when they are categorical (None
        when they are free, such as times), taken from the database itself and from the booking parameters."""
        schema: dict[str, dict[str, list[str] | None]] = {}
        for domain, rows in self.tables.items():
            slots: dict[str, list[str] | None] = {}
            for key in sorted({k for r in rows for k in r} - DESCRIPTIVE):
                values = sorted({normalize(r[key]) for r in rows if isinstance(r.get(key), (str, int, float))})
                if key == "name":
                    slots[key] = None
                elif len(values) > 1:  # a field with a single value does not distinguish entities
                    slots[key] = values if len(values) <= MAX_VALUES else None
            schema[domain] = slots
        for domain, parameters in self.booking.items():
            schema.setdefault(domain, {}).update({p: None for p in parameters if p not in schema.get(domain, {})})
        return {d: s for d, s in schema.items() if s}

    @staticmethod
    def _matches(row: dict, key: str, value: Any) -> bool:
        value = normalize(value)
        if value in EMPTY or key not in row:
            return True  # unknown keys, such as the booking parameters, do not filter
        stored = normalize(row[key])
        if key.lower() == "leaveat":
            return stored >= value
        if key.lower() == "arriveby":
            return stored <= value
        return stored == value or (key == "name" and (value in stored or stored in value))

    def find(self, domain: str, constraints: dict[str, Any]) -> list[dict]:
        rows = [r for r in self.tables.get(domain, [])
                if all(self._matches(r, k, v) for k, v in constraints.items())]
        if any(k.lower() == "leaveat" for k in constraints):
            rows.sort(key=lambda r: normalize(r.get("leaveAt", "")))
        elif any(k.lower() == "arriveby" for k in constraints):
            rows.sort(key=lambda r: normalize(r.get("arriveBy", "")), reverse=True)
        return rows

    @staticmethod
    def _summary(row: dict) -> dict:
        return {k: v for k, v in row.items() if k not in {"id", "location", "introduction", "signature"}}

    def search(self, domain: str, constraints: dict[str, Any], limit: int = 3) -> dict:
        """The number of entities of a domain that satisfy the constraints and the first ``limit`` of them."""
        rows = self.find(domain, constraints)
        return {"domain": domain, "count": len(rows), "entities": [self._summary(r) for r in rows[:limit]]}

    def book(self, domain: str, constraints: dict[str, Any]) -> dict:
        """A simulated booking: if an entity satisfies the constraints (or the domain is a service without entities,
        such as taxis), a reference made from the arguments, so that the same booking always gets the same one."""
        key = json.dumps([domain, sorted((k, str(v)) for k, v in constraints.items())])
        digest = hashlib.sha256(key.encode()).hexdigest()
        if domain in self.services:
            options = self.services[domain]
            pick = {k.removeprefix("taxi_").rstrip("s"): v[int(digest[i * 4:i * 4 + 4], 16) % len(v)]
                    for i, (k, v) in enumerate(options.items()) if v and not str(v[0]).startswith("^")}
            phone = "".join(str(int(c, 16) % 10) for c in digest[-10:])
            return {"domain": domain, "booked": True, "reference": digest[:8].upper(), "phone": phone, **pick}
        rows = self.find(domain, constraints)
        if not rows:
            return {"domain": domain, "booked": False, "reason": "no entity satisfies the constraints"}
        name = rows[0].get("name") or rows[0].get("trainID")
        return {"domain": domain, "booked": True, "name": name, "reference": digest[:8].upper()}

    def as_tool(self) -> Tool:
        return Tool(name="search", description="Search entities of a domain that satisfy the user constraints.",
                    parameters={"type": "object", "properties": {"domain": {"type": "string"},
                                                                 "constraints": {"type": "object"}},
                                "required": ["domain", "constraints"]},
                    function=lambda domain, constraints: self.search(domain, constraints))

    def as_booking_tool(self) -> Tool:
        return Tool(name="book", description="Book an entity of a domain that satisfies the user constraints.",
                    parameters={"type": "object", "properties": {"domain": {"type": "string"},
                                                                 "constraints": {"type": "object"}},
                                "required": ["domain", "constraints"]},
                    function=lambda domain, constraints: self.book(domain, constraints))
