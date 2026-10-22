"""Data types shared across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    """A dialogue turn."""

    speaker: str  # "user" or "system"
    text: str


@dataclass
class Action:
    """An induced action: a region of the Dialog2Flow space with a textual name."""

    id: int
    name: str


@dataclass
class Option:
    """An allowed action in the current state, with the probability assigned by the decider."""

    action: Action
    probability: float
    score: float


@dataclass
class Situation:
    """A situation retrieved from the memory: a user turn of another dialogue, the action the system took next and
    its response, together with where the situation comes from and how that dialogue ended."""

    user: str
    action: str
    response: str
    similarity: float
    provenance: str = "operator"  # "operator", "agent" or "correction"
    outcome: float | None = None
    dialogue_id: str | None = None


@dataclass
class ToolCall:
    """A call to a tool made by the agent during a turn."""

    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class TurnResult:
    """Everything the agent did in a turn: the response and the full trace of the decision."""

    text: str
    action: Action
    state: int
    options: list[Option]
    situations: list[Situation]
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Session:
    """The state of an ongoing conversation."""

    turns: list[Turn] = field(default_factory=list)
    states: list[int] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    constraints: dict[str, dict[str, str]] = field(default_factory=dict)
