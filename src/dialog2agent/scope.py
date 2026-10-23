"""Scope of a domain: whether a user turn resembles the turns of the dialogues from which the agent was built.

The check uses the representations of the agent itself, without another model. A user turn is within the scope of
the domain when its Dialog2Flow embedding is close to some induced user action and the memory holds a situation
similar to the turn on its own. Each signal has a threshold, a low percentile of the same signal over user turns of
the domain that were not used to build the artifact (see ``Scope.calibrate``), and the turn is out of scope when
either signal falls below its threshold. The thresholds are stored in the ``scope`` key of ``domain.json``.

Since the encoders are trained on dialogues in one language, turns in another language usually fall out of scope as
well, although the check does not identify the language.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .structure import normalize
from .types import Turn


@dataclass
class ScopeCheck:
    """Result of the check of a user turn: whether it is in scope and the value of each signal."""

    in_scope: bool
    action_similarity: float
    memory_similarity: float


@dataclass
class Scope:
    """Thresholds of the scope of a domain.

    Args:
        action_threshold: minimum cosine similarity between the turn and its closest induced user action.
        memory_threshold: minimum similarity between the turn on its own and its closest situation of the memory.
        language: language of the dialogues of the domain, for the clients of the agent.
        description: what the dialogues of the domain are about, for the clients of the agent.
        calibration: how the thresholds were obtained.
    """

    action_threshold: float
    memory_threshold: float
    language: str | None = None
    description: str | None = None
    calibration: str | None = None

    @staticmethod
    def signals(agent, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Similarity of each user turn, on its own, to its closest user action and to its closest situation."""
        e = agent.turn_encoder.encode(texts)
        action = (normalize(e) @ agent.structure.user_centroids.T).max(axis=1)
        queries = np.stack([agent.memory.represent(
            e[i], np.asarray(agent.dialogue_encoder.encode([Turn("user", text)], e[i:i + 1])).reshape(-1))
            for i, text in enumerate(texts)])
        # the memory is compared with blocks of turns, since one product per turn reads all the vectors each time
        memory = np.concatenate([(queries[i:i + 1024] @ agent.memory.vectors.T).max(axis=1)
                                 for i in range(0, len(texts), 1024)])
        return action, memory

    def check(self, agent, text: str) -> ScopeCheck:
        """Whether a user turn is within the scope of the domain."""
        action, memory = self.signals(agent, [text])
        return ScopeCheck(in_scope=bool(action[0] >= self.action_threshold and memory[0] >= self.memory_threshold),
                          action_similarity=float(action[0]), memory_similarity=float(memory[0]))

    @classmethod
    def calibrate(cls, agent, texts: list[str], percentile: float = 1.0, **kwargs) -> Scope:
        """Thresholds at the given percentile of each signal over user turns of the domain that were not used to
        build the artifact, such as those of its development dialogues."""
        action, memory = cls.signals(agent, texts)
        return cls(action_threshold=float(np.percentile(action, percentile)),
                   memory_threshold=float(np.percentile(memory, percentile)),
                   calibration=f"percentile {percentile:g} of each signal over {len(texts)} user turns", **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)
