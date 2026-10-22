"""TRACE and TRACE-Dialog in the transformers format.

TRACE is a BERT-style encoder whose inputs are not tokens but the embeddings of the turns of a dialogue, produced by
a base turn encoder (Dialog2Flow by default). It returns a contextual representation ``h_t`` of each turn; in the
autoregressive mode each turn attends only to past turns, so ``h_t`` can be computed during the conversation.
TRACE-Dialog adds learned [CLS] and [SEP] vectors in the input space and returns ``h_CLS``, a representation of the
whole dialogue.

The layers follow ``modeling_bert.py`` with three differences: continuous inputs projected to the hidden size,
speaker embeddings instead of token types, and positions that index turns.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel

from .configuration_trace import TraceConfig

CLS_ID, SEP_ID = 1, 2
ACT2FN = {"gelu": nn.GELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class TraceSelfAttention(nn.Module):
    def __init__(self, config: TraceConfig):
        super().__init__()
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.scaling = self.attention_head_size ** -0.5
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        shape = (*input_shape, -1, self.attention_head_size)
        q = self.query(hidden_states).view(shape).transpose(1, 2)
        k = self.key(hidden_states).view(shape).transpose(1, 2)
        v = self.value(hidden_states).view(shape).transpose(1, 2)
        out, _ = eager_attention_forward(self, q, k, v, attention_mask, self.scaling,
                                         dropout=self.dropout.p if self.training else 0.0)
        return out.reshape(*input_shape, -1).contiguous()


class TraceSelfOutput(nn.Module):
    def __init__(self, config: TraceConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)


class TraceAttention(nn.Module):
    def __init__(self, config: TraceConfig):
        super().__init__()
        self.self = TraceSelfAttention(config)
        self.output = TraceSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        return self.output(self.self(hidden_states, attention_mask), hidden_states)


class TraceIntermediate(nn.Module):
    def __init__(self, config: TraceConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))


class TraceOutput(nn.Module):
    def __init__(self, config: TraceConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)


class TraceLayer(nn.Module):
    def __init__(self, config: TraceConfig):
        super().__init__()
        self.attention = TraceAttention(config)
        self.intermediate = TraceIntermediate(config)
        self.output = TraceOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        return self.output(self.intermediate(attention_output), attention_output)


class TraceEncoder(nn.Module):
    def __init__(self, config: TraceConfig):
        super().__init__()
        self.layer = nn.ModuleList([TraceLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class TraceEmbeddings(nn.Module):
    """Turn embeddings projected to the hidden size, plus speaker and position embeddings."""

    def __init__(self, config: TraceConfig):
        super().__init__()
        self.input_proj = (nn.Linear(config.input_dim, config.hidden_size)
                           if config.input_dim != config.hidden_size else nn.Identity())
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.speaker_embeddings = (nn.Embedding(config.num_speakers, config.hidden_size)
                                   if config.use_speaker_embeddings else None)
        self.num_speakers = config.num_speakers
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds, speaker_ids=None):
        embeddings = self.input_proj(inputs_embeds)
        if self.speaker_embeddings is not None and speaker_ids is not None:
            embeddings = embeddings + self.speaker_embeddings(speaker_ids.clamp(min=0, max=self.num_speakers - 1))
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        return self.dropout(self.LayerNorm(embeddings + self.position_embeddings(positions)))


class TraceBody(nn.Module):
    """Embeddings and encoder (the counterpart of ``BertModel`` without pooler)."""

    def __init__(self, config: TraceConfig):
        super().__init__()
        self.config = config
        self.embeddings = TraceEmbeddings(config)
        self.encoder = TraceEncoder(config)

    def extended_mask(self, attention_mask, seq_len, dtype):
        allowed = attention_mask[:, None, None, :].bool()
        if self.config.is_causal:
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device))
            allowed = allowed & causal[None, None]
        mask = torch.zeros(allowed.shape, dtype=dtype, device=attention_mask.device)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def forward(self, inputs_embeds, attention_mask, speaker_ids=None):
        hidden = self.embeddings(inputs_embeds, speaker_ids)
        return self.encoder(hidden, self.extended_mask(attention_mask, inputs_embeds.shape[1], hidden.dtype))


class TracePreTrainedModel(PreTrainedModel):
    config_class = TraceConfig
    base_model_prefix = "bert"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ------------------------------------------------------------------ helpers shared by both models
    def speaker_ids(self, speakers: list[str]) -> list[int]:
        unknown = self.config.num_speakers - 1
        return [min(self.config.speaker_map.get(str(s).strip().lower(), unknown), unknown) for s in speakers]

    def base_embeddings(self, texts: list[str], base_encoder=None) -> np.ndarray:
        """Turn embeddings with the base encoder the model was trained on (loaded on first use)."""
        if base_encoder is None:
            if getattr(self, "_base", None) is None:
                from sentence_transformers import SentenceTransformer
                self._base = SentenceTransformer(self.config.base_encoder, device=str(self.device))
            base_encoder = self._base
        return np.asarray(base_encoder.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float32)


class TraceModel(TracePreTrainedModel):
    """TRACE: contextual representation ``h_t`` of every turn of a dialogue."""

    def __init__(self, config: TraceConfig):
        super().__init__(config)
        self.bert = TraceBody(config)
        self.output_proj = (nn.Linear(config.hidden_size, config.output_dim)
                            if config.hidden_size != config.output_dim else nn.Identity())
        self.post_init()

    def forward(self, inputs_embeds, attention_mask, speaker_ids=None):
        if inputs_embeds.shape[1] > self.config.max_turns:
            raise ValueError(f"{inputs_embeds.shape[1]} turns exceed max_turns ({self.config.max_turns})")
        return self.output_proj(self.bert(inputs_embeds, attention_mask, speaker_ids))

    @torch.no_grad()
    def encode_dialogue(self, turns: list[str], speakers: list[str], embeddings: np.ndarray | None = None,
                        base_encoder=None) -> np.ndarray:
        """``h_t`` of every turn, shape ``(n, output_dim)``. Dialogues longer than ``max_turns`` are split into
        consecutive windows. Pass ``embeddings`` to reuse precomputed base turn embeddings."""
        self.eval()
        e = embeddings if embeddings is not None else self.base_embeddings(turns, base_encoder)
        spk = self.speaker_ids(speakers)
        out = []
        for start in range(0, len(turns), self.config.max_turns):
            end = start + self.config.max_turns
            x = torch.as_tensor(e[start:end], dtype=self.dtype, device=self.device)[None]
            s = torch.as_tensor(spk[start:end], device=self.device)[None]
            att = torch.ones(x.shape[:2], dtype=torch.long, device=self.device)
            out.append(self(x, att, s if self.config.use_speaker_embeddings else None)[0].float().cpu().numpy())
        return np.concatenate(out)


class TraceDialogModel(TraceModel):
    """TRACE-Dialog: representation ``h_CLS`` of a whole dialogue, from [CLS] turns [SEP]."""

    def __init__(self, config: TraceConfig):
        super().__init__(config)
        self.special_embeddings = nn.Embedding(2, config.input_dim)
        self.post_init()

    def forward(self, inputs_embeds, attention_mask, speaker_ids=None, special_ids=None):
        if special_ids is not None:
            x = inputs_embeds.clone()
            x[special_ids == CLS_ID] = self.special_embeddings.weight[0].to(x.dtype)
            x[special_ids == SEP_ID] = self.special_embeddings.weight[1].to(x.dtype)
            inputs_embeds = x
        return super().forward(inputs_embeds, attention_mask, speaker_ids)

    @torch.no_grad()
    def encode_dialogue(self, turns: list[str], speakers: list[str], embeddings: np.ndarray | None = None,
                        base_encoder=None) -> np.ndarray:
        """``h_CLS`` of the dialogue, shape ``(output_dim,)``. Only the first ``max_turns - 2`` turns are used."""
        self.eval()
        e = embeddings if embeddings is not None else self.base_embeddings(turns, base_encoder)
        n = min(len(turns), self.config.max_turns - 2)
        x = torch.zeros(1, n + 2, self.config.input_dim, dtype=self.dtype, device=self.device)
        x[0, 1:n + 1] = torch.as_tensor(e[:n], dtype=self.dtype, device=self.device)
        spk = torch.full((1, n + 2), self.config.num_speakers - 1, dtype=torch.long, device=self.device)
        spk[0, 1:n + 1] = torch.as_tensor(self.speaker_ids(speakers[:n]), device=self.device)
        special = torch.zeros(1, n + 2, dtype=torch.long, device=self.device)
        special[0, 0], special[0, n + 1] = CLS_ID, SEP_ID
        att = torch.ones(1, n + 2, dtype=torch.long, device=self.device)
        h = self(x, att, spk if self.config.use_speaker_embeddings else None, special)
        return h[0, 0].float().cpu().numpy()
