"""TRACE-Dialog as a sentence-transformers model: a dialogue in, its embedding ``h_CLS`` out.

The input of TRACE-Dialog is not text but the sequence of base turn embeddings of a dialogue, so the default text
pipeline of sentence-transformers (tokenizer, transformer, pooling) does not apply. This module receives whole
dialogues, encodes their turns with the base encoder declared in the configuration and returns ``h_CLS`` as the
embedding of each dialogue. The weights are the same as with ``AutoModel``.

A dialogue is a list of chat messages (``role`` and ``content``) or of turns (``speaker`` and ``text``). The roles
``user`` and ``assistant`` (or ``system``) are the user and the system of task-oriented dialogue::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("jumafernandez/trace-dialog-deep-tod", trust_remote_code=True)
    dialogue = [{"role": "user", "content": "hi, I need a cheap restaurant in the centre"},
                {"role": "assistant", "content": "sure, what kind of food would you like?"}]
    embeddings = model.encode([dialogue])   # (1, 768)
"""
from __future__ import annotations

import torch
from torch import nn

from .modeling_trace import TraceDialogModel


def dialogue_turns(dialogue) -> tuple[list[str], list[str]]:
    """Texts and speakers of a dialogue given as chat messages or as turns."""
    texts, speakers = [], []
    for turn in dialogue:
        if not isinstance(turn, dict):
            raise TypeError("a dialogue is a list of dicts with 'role' and 'content' (or 'speaker' and 'text')")
        texts.append(str(turn.get("content", turn.get("text", "")) or ""))
        speakers.append(str(turn.get("role", turn.get("speaker", "user"))))
    return texts, speakers


class TraceDialogSentenceModule(nn.Module):
    """Input module of sentence-transformers that wraps TRACE-Dialog and its base turn encoder."""

    save_in_root = True

    def __init__(self, model: TraceDialogModel, base_encoder: str | None = None, batch_size: int = 64):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = model.eval()
        self.base = SentenceTransformer(base_encoder or model.config.base_encoder, device="cpu")
        self.batch_size = batch_size

    @property
    def modalities(self) -> list[str]:
        """Dialogues are passed as lists of chat messages."""
        return ["message"]

    def preprocess(self, inputs, prompt: str | None = None, **kwargs) -> dict:
        return {"dialogues": [dialogue_turns(d) for d in inputs]}

    def tokenize(self, inputs, **kwargs) -> dict:
        return self.preprocess(inputs)

    def forward(self, features: dict) -> dict:
        dialogues = features["dialogues"]
        texts = [t for turns, _ in dialogues for t in turns]
        e = self.base.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
        out, start = [], 0
        for turns, speakers in dialogues:
            out.append(torch.from_numpy(self.model.encode_dialogue(turns, speakers,
                                                                   embeddings=e[start:start + len(turns)])))
            start += len(turns)
        features["sentence_embedding"] = torch.stack(out).to(self.model.device)
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.model.config.output_dim

    get_embedding_dimension = get_sentence_embedding_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(cls, input_path: str) -> TraceDialogSentenceModule:
        return cls(TraceDialogModel.from_pretrained(input_path))
