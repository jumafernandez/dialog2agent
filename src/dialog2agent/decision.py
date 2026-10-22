"""Deciders: how the agent chooses the next action among the ones the structure allows.

The main decider scores every allowed action with the language model, as the log-probability of the output
``Action: <name>\\nResponse:`` after the prompt, and chooses the best one. Instead of generating a free phrase and
mapping it to an action, the decision is a choice among options defined in advance, which returns a distribution over
them and never leaves the allowed set. Simpler deciders based on the memory or on the structure are included as
references.
"""
from __future__ import annotations

from collections import Counter
from typing import Protocol

import numpy as np

from .backends import LLMBackend
from .prompting import action_prefix
from .structure import InducedStructure
from .types import Option, Situation


class Decider(Protocol):
    def decide(self, messages: list[dict], state: int, allowed: list[int],
               situations: list[Situation]) -> list[Option]:
        """Allowed actions with their probabilities, from the most to the least probable."""


def _options(structure: InducedStructure, allowed: list[int], scores: np.ndarray, scale: float) -> list[Option]:
    z = scale * (scores - scores.max())
    p = np.exp(z) / np.exp(z).sum()
    order = np.argsort(-scores, kind="stable")
    return [Option(action=structure.action(allowed[i]), probability=float(p[i]), score=float(scores[i]))
            for i in order]


class LLMScoringDecider:
    """Scores the allowed actions with the language model.

    Actions that share a name receive the same score, and the tie is broken by the probability of the action in the
    state. ``scale`` is a calibration factor of the probabilities (it does not change the chosen action).
    """

    def __init__(self, backend: LLMBackend, structure: InducedStructure, scale: float = 1.0):
        self.backend, self.structure, self.scale = backend, structure, scale

    def decide(self, messages, state, allowed, situations):
        names = list(dict.fromkeys(self.structure.names[a] for a in allowed))
        by_name = dict(zip(names, self.backend.score(messages, [action_prefix(n) for n in names])))
        prior = self.structure.prior(state)
        scores = np.array([by_name[self.structure.names[a]] + 1e-6 * prior[a] for a in allowed])
        return _options(self.structure, allowed, scores, self.scale)


class MemoryVoteDecider:
    """Chooses the action name most frequent among the retrieved situations."""

    def __init__(self, structure: InducedStructure):
        self.structure = structure

    def decide(self, messages, state, allowed, situations):
        votes = Counter(s.action for s in situations)
        prior = self.structure.prior(state)
        scores = np.array([votes.get(self.structure.names[a], 0) + 1e-6 * prior[a] for a in allowed], float)
        return _options(self.structure, allowed, scores, scale=1.0)


class StructureDecider:
    """Chooses the most probable action of the state according to the structure."""

    def __init__(self, structure: InducedStructure):
        self.structure = structure

    def decide(self, messages, state, allowed, situations):
        prior = self.structure.prior(state)
        scores = np.log(np.array([prior[a] for a in allowed]) + 1e-12)
        return _options(self.structure, allowed, scores, scale=1.0)
