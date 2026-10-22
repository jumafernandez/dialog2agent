"""Turn and dialogue representations.

- ``e_t``: Dialog2Flow embedding of each turn, which characterizes the action the turn performs.
- ``h_t``: TRACE contextual representation of each turn (autoregressive mode, only past turns), used for the states.
- ``h_CLS``: TRACE-Dialog representation of the dialogue up to the current turn, used for the memory.

All of them can be computed during the conversation, since none needs future turns.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import Turn

DIALOG2FLOW = "sergioburdisso/dialog2flow-joint-bert-base"


class TurnEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Embeddings of a list of turns, shape ``(n, d)``."""


class ContextEncoder(Protocol):
    def encode(self, turns: list[Turn], e: np.ndarray) -> np.ndarray:
        """Contextual representation ``h_t`` of every turn of a dialogue from its turn embeddings."""


class DialogueEncoder(Protocol):
    def encode(self, turns: list[Turn], e: np.ndarray) -> np.ndarray:
        """Representation ``h_CLS`` of the dialogue made of ``turns``."""


class Dialog2FlowEncoder:
    """Dialog2Flow turn encoder (Burdisso et al., 2024)."""

    def __init__(self, model: str = DIALOG2FLOW, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, show_progress_bar=False,
                                 convert_to_numpy=True).astype(np.float32)


class _HubModel:
    """A TRACE-family model loaded from the Hugging Face Hub with its own code (``trust_remote_code``).

    Pin ``revision`` to a commit so that an update of the model repository cannot change the behavior of the agent.
    """

    def __init__(self, checkpoint: str, device: str | None = None, revision: str | None = None):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision=revision).eval()
        if device:
            self.model.to(device)


class TraceEncoder(_HubModel):
    """TRACE contextual turn encoder: ``h_t`` of every turn (autoregressive mode, only past turns)."""

    def encode(self, turns: list[Turn], e: np.ndarray) -> np.ndarray:
        return self.model.encode_dialogue([t.text for t in turns], [t.speaker for t in turns], embeddings=e)


class TraceDialogEncoder(_HubModel):
    """TRACE-Dialog dialogue encoder: ``h_CLS`` of the dialogue up to the last turn."""

    def encode(self, turns: list[Turn], e: np.ndarray) -> np.ndarray:
        return self.model.encode_dialogue([t.text for t in turns], [t.speaker for t in turns], embeddings=e)
