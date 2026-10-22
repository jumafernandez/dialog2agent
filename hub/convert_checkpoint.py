"""Convert a TRACE or TRACE-Dialog checkpoint of the training code into a Hugging Face model with remote code.

The training checkpoints store ``config.json`` (the training ``ModelConfig``) and ``model.safetensors``. The output
directory contains the transformers configuration, the weights without the training heads, and the two code files,
so that the model loads with ``AutoModel.from_pretrained(path, trust_remote_code=True)``.

Usage:
    python hub/convert_checkpoint.py CHECKPOINT_DIR OUTPUT_DIR --kind trace|trace-dialog \
        [--base-encoder sergioburdisso/dialog2flow-joint-bert-base]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_hf.configuration_trace import TraceConfig
from trace_hf.modeling_trace import TraceDialogModel, TraceModel

TRAINING_ONLY = ("reconstruction_head.", "next_turn_head.", "mask_embedding")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--kind", choices=["trace", "trace-dialog"], required=True)
    ap.add_argument("--base-encoder", default="sergioburdisso/dialog2flow-joint-bert-base")
    a = ap.parse_args()

    m = json.loads((a.checkpoint / "config.json").read_text())
    assert m.get("arch", "v1") == "v2", "only the BERT-faithful (v2) architecture is supported"
    config = TraceConfig(base_encoder=a.base_encoder, input_dim=m["input_dim"], hidden_size=m["hidden_dim"],
                         num_hidden_layers=m["num_layers"], num_attention_heads=m["num_heads"],
                         intermediate_size=m.get("ff_dim") or 4 * m["hidden_dim"], hidden_act=m["activation"],
                         hidden_dropout_prob=m["dropout"], attention_probs_dropout_prob=m["dropout"],
                         max_turns=m["max_turns"], is_causal=m["attention_mode"] == "autoregressive",
                         use_speaker_embeddings=m["use_speaker_embeddings"], num_speakers=m["num_speakers"],
                         output_dim=m.get("output_dim"), with_special_tokens=a.kind == "trace-dialog")
    cls = TraceDialogModel if a.kind == "trace-dialog" else TraceModel
    TraceConfig.register_for_auto_class()
    cls.register_for_auto_class("AutoModel")
    model = cls(config)
    state = {k: v for k, v in load_file(str(a.checkpoint / "model.safetensors")).items()
             if not k.startswith(TRAINING_ONLY)}
    model.load_state_dict(state, strict=True)
    model.save_pretrained(a.output)
    print("CONVERTED", a.kind, a.output, sum(p.numel() for p in model.parameters()), "parameters")


if __name__ == "__main__":
    main()
