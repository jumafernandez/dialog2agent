"""Check that TRACE-Dialog loaded with sentence-transformers reproduces the representations of the training code.

Dialogue prefixes of a corpus already encoded with the training code are passed as chat messages, so the base turn
embeddings are recomputed from the text and the comparison covers the whole pipeline from text to ``h_CLS``. The
script also checks that the turn format (``speaker`` and ``text``) gives the same embeddings and that the
sentence-transformers module agrees with ``AutoModel``.

Usage:
    python hub/verify_sentence_transformers.py --model DIR_OR_REPO --turns TURNS.pkl --dialog-h TDLG.npy \
        [--prefixes 500]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import sentence_transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModel


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--dialog-h", required=True)
    ap.add_argument("--prefixes", type=int, default=500)
    ap.add_argument("--revision")
    a = ap.parse_args()

    t = pd.read_pickle(a.turns).sort_values(["dialogue_id", "turn_id"])
    cls_ref = np.load(a.dialog_h, mmap_mode="r")
    by_dialogue = {d: g for d, g in t.groupby("dialogue_id", sort=False)}
    picks = t.sample(a.prefixes, random_state=0)
    dialogues, turn_format, rows = [], [], []
    for d, k in zip(picks["dialogue_id"], picks["turn_id"]):
        g = by_dialogue[d]
        g = g[g["turn_id"] <= k]
        texts, speakers = g["text"].fillna("").tolist(), g["speaker"].tolist()
        dialogues.append([{"role": "user" if s == "user" else "assistant", "content": x}
                          for s, x in zip(speakers, texts)])
        turn_format.append([{"speaker": s, "text": x} for s, x in zip(speakers, texts)])
        rows.append(g.get("row_id", g.index).to_numpy()[-1])

    model = SentenceTransformer(a.model, trust_remote_code=True, revision=a.revision)
    emb = model.encode(dialogues, batch_size=16)
    ref = np.asarray(cls_ref[np.array(rows)])
    emb_turns = model.encode(turn_format, batch_size=16)
    auto = AutoModel.from_pretrained(a.model, trust_remote_code=True, revision=a.revision).eval()
    n_auto = min(50, len(dialogues))
    emb_auto = np.stack([auto.encode_dialogue([m["content"] for m in d], [m["role"] for m in d])
                         for d in dialogues[:n_auto]])
    lengths = [len(d) for d in dialogues]

    diff, cos = np.abs(emb - ref).max(), cosine(emb, ref).min()
    print(f"sentence-transformers {sentence_transformers.__version__}, embedding {emb.shape}, "
          f"prefixes of {min(lengths)} to {max(lengths)} turns")
    print(f"vs training code: max |diff| {diff:.2e}, min cosine {cos:.6f} ({len(rows)} prefixes)")
    print(f"turn format vs messages: max |diff| {np.abs(emb_turns - emb).max():.2e}")
    print(f"vs AutoModel: max |diff| {np.abs(emb_auto - emb[:n_auto]).max():.2e} ({n_auto} prefixes)")
    # the turn format sorts the batches differently, so the base embeddings may differ at the level of float32
    ok = diff < 1e-3 and np.abs(emb_turns - emb).max() < 1e-4 and np.abs(emb_auto - emb[:n_auto]).max() < 1e-4
    print("VERIFY-ST-OK" if ok else "VERIFY-ST-DIFF")


if __name__ == "__main__":
    main()
