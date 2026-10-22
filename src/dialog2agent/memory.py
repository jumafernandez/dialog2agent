"""Conversational memory: situations of other dialogues retrieved by similarity.

Each entry is a user turn of a stored dialogue, represented by ``[e_t ; lambda * h_CLS]``, that is, the action of the
turn (Dialog2Flow) combined with the representation of the dialogue up to that turn (TRACE-Dialog), together with the
action the system took next, its response, where the entry comes from and how its dialogue ended. Retrieval can be
plain or conditioned by the structure (entries in the same state or whose next action is allowed).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .structure import normalize
from .types import Situation


class SituationMemory:
    """Store of situations with exact cosine search."""

    def __init__(self, vectors: np.ndarray | None = None, entries: list[dict] | None = None,
                 context_weight: float = 0.75):
        self.vectors = normalize(vectors) if vectors is not None else np.zeros((0, 0), np.float32)
        self.entries = list(entries or [])
        self.context_weight = context_weight
        self._columns: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def _index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """State, next action and dialogue of every entry as arrays (-1 when missing), rebuilt after additions."""
        if self._columns is None or len(self._columns[0]) != len(self.entries):
            def column(key):
                return np.array([-1 if e.get(key) is None else e[key] for e in self.entries], dtype=np.int64)
            self._columns = (column("state"), column("action_id"),
                             np.array([e.get("dialogue_id") for e in self.entries], dtype=object))
        return self._columns

    def represent(self, e: np.ndarray, h_cls: np.ndarray) -> np.ndarray:
        """Representation of a situation from the turn (``e``) and dialogue (``h_cls``) embeddings."""
        return normalize(np.concatenate([normalize(e), self.context_weight * normalize(h_cls)], axis=-1))

    def search(self, query: np.ndarray, k: int = 10, state: int | None = None, allowed: list[int] | None = None,
               exclude_dialogue: str | None = None) -> list[Situation]:
        """The ``k`` most similar situations. With ``state`` and ``allowed``, the search is restricted to entries in
        the same state or whose next action is allowed, and falls back to the whole memory when fewer than ``k``
        entries qualify."""
        if not self.entries:
            return []
        states, actions, dialogues = self._index()
        candidates = np.arange(len(self.entries))
        if exclude_dialogue is not None:
            candidates = candidates[dialogues != exclude_dialogue]
        if state is not None or allowed is not None:
            same = states[candidates] == (-2 if state is None else state)
            cond = candidates[same | np.isin(actions[candidates], list(allowed or []))]
            if len(cond) >= k:
                candidates = cond
        sims = self.vectors[candidates] @ normalize(query)
        order = np.argsort(-sims, kind="stable")[:k]
        out = []
        for i, sim in zip(candidates[order], sims[order]):
            entry = self.entries[i]
            out.append(Situation(user=entry["user"], action=entry["action"], response=entry["response"],
                                 similarity=float(sim), provenance=entry.get("provenance", "operator"),
                                 outcome=entry.get("outcome"), dialogue_id=entry.get("dialogue_id")))
        return out

    def add(self, vector: np.ndarray, situation: Situation, state: int | None = None,
            action_id: int | None = None) -> None:
        """Add an experience, for example a turn of the agent itself or a correction of the user."""
        row = normalize(vector[None])
        self.vectors = row if self.vectors.size == 0 else np.concatenate([self.vectors, row])
        self.entries.append({**asdict(situation), "state": state, "action_id": action_id})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with open(path / "entries.jsonl", "w") as f:
            f.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in self.entries)
        (path / "memory.json").write_text(json.dumps({"context_weight": self.context_weight}))

    @classmethod
    def load(cls, path: str | Path) -> SituationMemory:
        path = Path(path)
        with open(path / "entries.jsonl") as f:
            entries = [json.loads(line) for line in f]
        meta = json.loads((path / "memory.json").read_text())
        return cls(np.load(path / "vectors.npy"), entries, **meta)
