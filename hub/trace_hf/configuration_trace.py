"""Configuration of TRACE (contextual turn representations) and TRACE-Dialog (dialogue representations)."""
from __future__ import annotations

try:
    from transformers import PreTrainedConfig as _BaseConfig
except ImportError:  # transformers < 5
    from transformers import PretrainedConfig as _BaseConfig

DEFAULT_SPEAKER_MAP = {"user": 0, "customer": 0, "system": 1, "assistant": 1, "bot": 1, "agent": 2, "operator": 2}


class TraceConfig(_BaseConfig):
    """BERT-style encoder over a sequence of turn embeddings.

    Args:
        base_encoder: sentence-transformers model that produces the turn embeddings the encoder was trained on.
            TRACE is agnostic to the base: a checkpoint trained over another base declares it here.
        input_dim: dimension of the base turn embeddings.
        hidden_size, num_hidden_layers, num_attention_heads, intermediate_size: encoder size.
        max_turns: maximum number of positions (turns, plus [CLS] and [SEP] in TRACE-Dialog).
        is_causal: autoregressive attention (each turn attends only to past turns).
        num_speakers: size of the speaker vocabulary; unknown speakers use the last id.
        with_special_tokens: TRACE-Dialog, with learned [CLS] and [SEP] vectors in the input space.
    """

    model_type = "trace"

    def __init__(self, base_encoder: str = "sergioburdisso/dialog2flow-joint-bert-base", input_dim: int = 768,
                 hidden_size: int = 768, num_hidden_layers: int = 4, num_attention_heads: int = 8,
                 intermediate_size: int | None = None, hidden_act: str = "gelu", hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1, max_turns: int = 64, is_causal: bool = False,
                 use_speaker_embeddings: bool = True, num_speakers: int = 4, output_dim: int | None = None,
                 layer_norm_eps: float = 1e-12, initializer_range: float = 0.02, with_special_tokens: bool = False,
                 speaker_map: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.base_encoder = base_encoder
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_turns = max_turns
        self.max_position_embeddings = max_turns
        self.is_causal = is_causal
        self.use_speaker_embeddings = use_speaker_embeddings
        self.num_speakers = num_speakers
        self.output_dim = output_dim or input_dim
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.with_special_tokens = with_special_tokens
        self.speaker_map = speaker_map or dict(DEFAULT_SPEAKER_MAP)
