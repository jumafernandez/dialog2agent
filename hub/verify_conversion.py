"""Check that the converted models reproduce the representations computed with the training code.

Compares, on dialogues of a corpus already encoded with the training code, the ``h_t`` of TRACE and the ``h_CLS``
of TRACE-Dialog (one per prefix of each dialogue) produced by the converted models loaded with
``AutoModel.from_pretrained(..., trust_remote_code=True)``.

Usage:
    python hub/verify_conversion.py --turns TURNS.pkl --e D2F.npy --trace HF_TRACE --trace-h TRACE.npy \
        --dialog HF_TRACE_DIALOG --dialog-h TDLG.npy [--dialogues 50]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from transformers import AutoModel


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--e", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--trace-h", required=True)
    ap.add_argument("--dialog", required=True)
    ap.add_argument("--dialog-h", required=True)
    ap.add_argument("--dialogues", type=int, default=50)
    a = ap.parse_args()

    t = pd.read_pickle(a.turns)
    e, h_ref, cls_ref = np.load(a.e), np.load(a.trace_h, mmap_mode="r"), np.load(a.dialog_h, mmap_mode="r")
    trace = AutoModel.from_pretrained(a.trace, trust_remote_code=True).eval()
    dialog = AutoModel.from_pretrained(a.dialog, trust_remote_code=True).eval()
    ids = t["dialogue_id"].drop_duplicates().sample(a.dialogues, random_state=0)
    dh, ch, cos_h, cos_c = [], [], [], []
    for d in ids:
        g = t[t["dialogue_id"] == d].sort_values("turn_id")
        rows, spk = g.index.to_numpy(), g["speaker"].tolist()
        h = trace.encode_dialogue([""] * len(rows), spk, embeddings=e[rows])
        dh.append(np.abs(h - h_ref[rows]).max())
        cos_h.append(cosine(h, h_ref[rows]).min())
        for i in range(len(rows)):
            c = dialog.encode_dialogue([""] * (i + 1), spk[:i + 1], embeddings=e[rows[:i + 1]])
            ch.append(np.abs(c - cls_ref[rows[i]]).max())
            cos_c.append(cosine(c, cls_ref[rows[i]]))
    print(f"TRACE h_t: max |diff| {max(dh):.2e}, min cosine {min(cos_h):.6f} ({a.dialogues} dialogues)")
    print(f"TRACE-Dialog h_CLS: max |diff| {max(ch):.2e}, min cosine {min(cos_c):.6f} ({len(ch)} prefixes)")
    print("VERIFY-OK" if max(dh) < 1e-3 and max(ch) < 1e-3 else "VERIFY-DIFF")


if __name__ == "__main__":
    main()
