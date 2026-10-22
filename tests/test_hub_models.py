import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hub"))
from trace_hf.configuration_trace import TraceConfig
from trace_hf.modeling_trace import TraceDialogModel, TraceModel


def tiny(**kw):
    return TraceConfig(input_dim=12, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, max_turns=8, **kw)


def test_trace_is_causal_and_windowed():
    model = TraceModel(tiny(is_causal=True)).eval()
    e = np.random.default_rng(0).normal(size=(11, 12)).astype(np.float32)
    speakers = ["user", "system"] * 5 + ["user"]
    h = model.encode_dialogue([""] * 11, speakers, embeddings=e)
    assert h.shape == (11, 12)
    h_prefix = model.encode_dialogue([""] * 5, speakers[:5], embeddings=e[:5])
    assert np.allclose(h[:5], h_prefix, atol=1e-5)  # past turns do not see the future


def test_trace_dialog_roundtrip_with_remote_code(tmp_path):
    TraceConfig.register_for_auto_class()
    TraceDialogModel.register_for_auto_class("AutoModel")
    model = TraceDialogModel(tiny(with_special_tokens=True)).eval()
    model.save_pretrained(tmp_path)
    assert (tmp_path / "modeling_trace.py").exists() and (tmp_path / "configuration_trace.py").exists()
    loaded = transformers.AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    e = np.random.default_rng(1).normal(size=(4, 12)).astype(np.float32)
    a = model.encode_dialogue([""] * 4, ["user", "system", "user", "system"], embeddings=e)
    b = loaded.encode_dialogue([""] * 4, ["user", "system", "user", "system"], embeddings=e)
    assert a.shape == (12,) and np.allclose(a, b, atol=1e-6)
