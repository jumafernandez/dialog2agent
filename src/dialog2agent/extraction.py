"""Constraints of the user for the tools, written by the language model from the dialogue.

When the decided action calls a tool, the model reads the dialogue and the schema of the database (the domains, their
slots and the values of the categorical slots, all taken from the database itself) and writes the constraints that
the user has stated as a JSON object. The output is checked against the schema: unknown domains and slots are dropped
and categorical values must be one of the values of the database.
"""
from __future__ import annotations

import json
import re

from .tools import EMPTY, normalize
from .types import Turn

INSTRUCTION = ("You read a task-oriented dialogue between a user and the system of a town information centre and "
               "write the constraints that the user has stated so far, which the system needs to search its "
               "database or make a booking. Answer only with a JSON object that maps each service to an object of "
               "slots and values, using only the services, slots and values listed below. Leave out the slots that "
               "the user has not stated, write times as HH:MM, and write first the service of the user's current "
               "request.\n\nServices and slots:\n{schema}")
MAX_CONTEXT_TURNS = 24


def describe(schema: dict[str, dict[str, list[str] | None]]) -> str:
    lines = []
    for domain, slots in schema.items():
        parts = [f"{slot} ({', '.join(values)})" if values else slot for slot, values in slots.items()]
        lines.append(f"- {domain}: " + "; ".join(parts))
    return "\n".join(lines)


def key(slot: str) -> str:
    return re.sub(r"[^a-z]", "", str(slot).lower())


def first_object(text: str) -> str | None:
    """The first balanced JSON object of a text."""
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)) if start >= 0 else []:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
    return None


class ConstraintExtractor:
    """Writes the constraints of the user with the language model of the agent."""

    def __init__(self, backend, schema: dict[str, dict[str, list[str] | None]], max_new_tokens: int = 160):
        self.backend, self.schema, self.max_new_tokens = backend, schema, max_new_tokens
        self.instruction = INSTRUCTION.format(schema=describe(schema))

    def parse(self, text: str) -> dict[str, dict[str, str]]:
        """The constraints of a generated text, checked against the schema."""
        found = first_object(text)
        try:
            data = json.loads(found) if found else {}
        except json.JSONDecodeError:
            return {}
        constraints: dict[str, dict[str, str]] = {}
        for domain, slots in (data.items() if isinstance(data, dict) else []):
            domain = normalize(domain)
            if domain not in self.schema or not isinstance(slots, dict):
                continue
            names = {key(n): n for n in self.schema[domain]}  # "leave at" and "leave_at" are the slot leaveAt
            constraints.setdefault(domain, {})  # the service is kept even without valid slots, as for a taxi
            for slot, value in slots.items():
                slot = names.get(key(slot), slot)
                allowed = self.schema[domain].get(slot, "unknown")
                value = normalize(value)
                if allowed == "unknown" or value in EMPTY or (allowed is not None and value not in allowed):
                    continue
                constraints.setdefault(domain, {})[slot] = value
        return constraints

    def extract(self, turns: list[Turn]) -> dict[str, dict[str, str]]:
        context = list(turns[-MAX_CONTEXT_TURNS:])
        while context and context[0].speaker != "user":
            context = context[1:]
        chat = [{"role": "system", "content": self.instruction}] + [
            {"role": "user" if t.speaker == "user" else "assistant", "content": t.text} for t in context]
        return self.parse("{" + self.backend.generate(chat, prefix="{", max_new_tokens=self.max_new_tokens))
