"""Language model backends.

The agent needs two operations from the language model: scoring a set of continuations of a prompt (to decide the
action among the allowed ones) and generating a continuation (to write the response). Any backend that provides both
can be used, from a local model with transformers to an inference server.
"""
from __future__ import annotations

from typing import Protocol

import torch


class LLMBackend(Protocol):
    def score(self, messages: list[dict], continuations: list[str]) -> list[float]:
        """Sum of the log-probabilities of each continuation after the chat prompt."""

    def generate(self, messages: list[dict], prefix: str = "", max_new_tokens: int = 120) -> str:
        """Greedy continuation of the chat prompt, starting after ``prefix``."""


def _fit(messages: list[dict], count, budget: int) -> list[dict]:
    """Drop the oldest dialogue turns (never the system message) until the prompt fits in ``budget`` tokens, so that
    the remaining dialogue starts with a user turn."""
    system, turns = messages[:1], list(messages[1:])
    while len(turns) > 1 and count(system + turns) > budget:
        turns = turns[1:]
        while turns and turns[0]["role"] != "user":
            turns = turns[1:]
    return system + turns


class TransformersBackend:
    """A local causal language model, optionally with a LoRA adapter, run with transformers."""

    def __init__(self, model, tokenizer, max_tokens: int = 2048, batch_size: int = 8):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.device = next(model.parameters()).device
        eos = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
        self.eos = sorted(t for t in eos if t is not None and t >= 0)
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos[0]

    @classmethod
    def from_pretrained(cls, model: str, adapter: str | None = None, device: str | None = None,
                        dtype: torch.dtype = torch.bfloat16, **kwargs) -> TransformersBackend:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and \
                torch.xpu.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.padding_side = "left"
        lm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(device)
        if adapter:
            from peft import PeftModel
            lm = PeftModel.from_pretrained(lm, adapter)
        return cls(lm, tokenizer, **kwargs)

    def _template(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _prompt(self, messages: list[dict], reserve: int) -> list[int]:
        fitted = _fit(messages, lambda m: len(self._ids(self._template(m))), self.max_tokens - reserve)
        return self._ids(self._template(fitted))

    @torch.no_grad()
    def score(self, messages: list[dict], continuations: list[str]) -> list[float]:
        outs = [self._ids(c) for c in continuations]
        p = self._prompt(messages, reserve=max(max(map(len, outs)), 120))
        scores: list[float] = []
        for s in range(0, len(outs), self.batch_size):
            batch = outs[s:s + self.batch_size]
            width = max(map(len, batch))
            ids = torch.full((len(batch), len(p) + width), self.pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for j, o in enumerate(batch):
                ids[j, :len(p) + len(o)] = torch.tensor(p + o)
                att[j, :len(p) + len(o)] = 1
            pos = torch.arange(len(p) - 1, len(p) + width - 1)
            logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                                logits_to_keep=pos.to(self.device)).logits.float()
            lp = torch.log_softmax(logits, -1).cpu()
            token_lp = lp.gather(-1, ids[:, len(p):].unsqueeze(-1)).squeeze(-1)
            mask = torch.stack([torch.arange(width) < len(o) for o in batch])
            scores += (token_lp * mask).sum(1).tolist()
        return scores

    @torch.no_grad()
    def generate(self, messages: list[dict], prefix: str = "", max_new_tokens: int = 120) -> str:
        p = self._prompt(messages, reserve=max_new_tokens + len(self._ids(prefix)))
        ids = torch.tensor([p + self._ids(prefix)], device=self.device)
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.pad, eos_token_id=self.eos)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


class OpenAICompatibleBackend:
    """A model served by an OpenAI-compatible inference server, such as vLLM.

    Scoring requires the log-probabilities of the prompt tokens (``prompt_logprobs`` in vLLM), which is what allows
    the web demo to run the same decision as the evaluated agent.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        self.base_url, self.model, self.api_key = base_url, model, api_key

    def score(self, messages: list[dict], continuations: list[str]) -> list[float]:
        raise NotImplementedError("scoring through an inference server is planned for the web demo")

    def generate(self, messages: list[dict], prefix: str = "", max_new_tokens: int = 120) -> str:
        raise NotImplementedError("generation through an inference server is planned for the web demo")
