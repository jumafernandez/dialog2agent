"""The induced structure: actions, states and the actions allowed in each state.

Actions are regions of the Dialog2Flow space of turn embeddings, obtained by cutting an agglomerative clustering of
the turns of each speaker. States are regions of the combined representation ``[e_t ; lambda * h_t]`` of user turns,
where ``h_t`` is the contextual representation of TRACE. The set of allowed actions of a state contains the system
actions that followed that state in the training dialogues with an estimated probability of at least ``tau``.
None of these steps uses dialogue annotations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import Action


def normalize(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@dataclass
class InducedStructure:
    """Induced automaton of states and actions.

    Attributes:
        system_centroids: unit-norm centroids of the system actions in the Dialog2Flow space, shape ``(A, d)``.
        user_centroids: unit-norm centroids of the user actions, shape ``(U, d)``.
        state_centers: centers of the states in the combined space, shape ``(S, d')``.
        transitions: counts of the system action that followed each state in the training dialogues, shape ``(S, A)``.
        names: textual name of each system action.
        tau: minimum probability of an allowed action.
        context_weight: weight ``lambda`` of the contextual representation in the state space.
    """

    system_centroids: np.ndarray
    user_centroids: np.ndarray
    state_centers: np.ndarray
    transitions: np.ndarray
    names: list[str]
    tau: float = 0.01
    context_weight: float = 0.75

    @property
    def num_actions(self) -> int:
        return len(self.system_centroids)

    def action(self, i: int) -> Action:
        return Action(id=int(i), name=self.names[i])

    def system_action(self, e: np.ndarray) -> np.ndarray:
        """Closest system action of each Dialog2Flow embedding."""
        return np.argmax(normalize(e) @ self.system_centroids.T, axis=-1)

    def user_action(self, e: np.ndarray) -> np.ndarray:
        """Closest user action of each Dialog2Flow embedding."""
        return np.argmax(normalize(e) @ self.user_centroids.T, axis=-1)

    def state(self, e: np.ndarray, h: np.ndarray) -> np.ndarray:
        """State of user turns from their Dialog2Flow (``e``) and TRACE (``h``) embeddings."""
        x = normalize(np.concatenate([normalize(e), self.context_weight * normalize(h)], axis=-1))
        c = self.state_centers.astype(np.float32)
        # squared Euclidean distance without the term of x, which is the same for every center
        return np.argmin((c * c).sum(-1) - 2 * x @ c.T, axis=-1)

    def prior(self, state: int) -> np.ndarray:
        """Probability of each system action in a state, P(a | s), with the global distribution as fallback."""
        c = self.transitions[state]
        if c.sum() == 0:
            c = self.transitions.sum(0)
        return c / c.sum()

    def allowed(self, state: int) -> list[int]:
        """Allowed actions of a state, from the most to the least probable."""
        p = self.prior(state)
        order = np.argsort(-p, kind="stable")
        keep = [int(a) for a in order if p[a] >= self.tau]
        return keep or [int(order[0])]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "structure.npz", system_centroids=self.system_centroids, user_centroids=self.user_centroids,
                 state_centers=self.state_centers, transitions=self.transitions)
        (path / "structure.json").write_text(json.dumps(
            {"names": self.names, "tau": self.tau, "context_weight": self.context_weight}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> InducedStructure:
        path = Path(path)
        arrays = np.load(path / "structure.npz")
        meta = json.loads((path / "structure.json").read_text())
        return cls(system_centroids=arrays["system_centroids"], user_centroids=arrays["user_centroids"],
                   state_centers=arrays["state_centers"], transitions=arrays["transitions"], **meta)
