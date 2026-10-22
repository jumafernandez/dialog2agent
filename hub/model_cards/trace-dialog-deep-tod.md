---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: feature-extraction
base_model: sergioburdisso/dialog2flow-joint-bert-base
tags:
  - dialogue
  - task-oriented-dialogue
  - dialogue-embeddings
---

# TRACE-Dialog (deep, task-oriented, supervised)

TRACE-Dialog produces a single embedding of a whole dialogue. It extends the TRACE contextual turn encoder with a
learned `[CLS]` vector, placed before the turns and followed by a learned `[SEP]` vector, and the representation of
the dialogue is the output $h_{[\mathrm{CLS}]}$ at that position. The slot is first trained without an objective of
its own, following RoBERTa, on dialogues packed without cross-contamination, and then with a supervised similarity
stage in which dialogues that share the same set of services are positives. The representation is sensitive to the
trajectory of the dialogue, that is, to the order of its turns.

## Model details

| | |
|---|---|
| Base turn encoder (frozen) | [`sergioburdisso/dialog2flow-joint-bert-base`](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base) |
| Architecture | BERT-style encoder over turn embeddings with `[CLS]` and `[SEP]` vectors: 12 layers, 12 heads, hidden size 768 |
| Parameters | 85.1M |
| Attention | bidirectional |
| Inputs | turn embeddings (768) and speaker ids, up to 62 turns |
| Output | $h_{[\mathrm{CLS}]} \in \mathbb{R}^{768}$ for the dialogue |
| Training data | task-oriented dialogue collection of Dialog2Flow; similarity stage supervised with the service sets of SGD |
| Training (similarity stage) | 5 epochs, learning rate $2 \times 10^{-5}$, temperature 0.05, seed 42 |

## Usage

The model ships its own code, so it loads with `trust_remote_code=True`. Pinning a `revision` is recommended.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("jumafernandez/trace-dialog-deep-tod", trust_remote_code=True)

turns = ["hi, I need a cheap restaurant in the centre", "sure, what kind of food would you like?", "italian"]
speakers = ["user", "system", "user"]
h_cls = model.encode_dialogue(turns, speakers)   # (768,)
```

`encode_dialogue` computes the base embeddings with the base encoder declared in the configuration, which requires `sentence-transformers`. Precomputed base
embeddings can be passed with `embeddings=`. Only the first 62 turns are used.

## Evaluation

In the retrieval of conversational situations on SpokenWOZ, where the next dialogue act is predicted by the vote of
the continuations of the $k$ nearest turns (1,019 single-domain training dialogues, cross-validated), the dialogue
representation alone summarizes the services of the conversation rather than its last turn, and it is most useful
combined with the embedding of the current turn:

| Representation | Accuracy | Macro-F1 | Coverage@5 |
|---|---|---|---|
| $e_t$ (Dialog2Flow) | 0.454 | 0.135 | 0.646 |
| $h_{[\mathrm{CLS}]}$ of the prefix (this model) | 0.301 | 0.084 | 0.548 |
| $[e_t ; \lambda h_{[\mathrm{CLS}]}]$ | 0.482 | 0.173 | 0.685 |

This combination is the representation of the conversational memory of the
[dialog2agent](https://github.com/jumafernandez/dialog2agent) agent.

This repository contains the checkpoint converted to the transformers format. The conversion was verified against
the representations computed with the training code on 1,882 dialogue prefixes of SpokenWOZ, with a maximum
absolute difference of $7.4 \times 10^{-6}$.

## Limitations

The model was trained on English task-oriented dialogue and expects embeddings of the base encoder it was trained
on. The similarity stage relies on the services of SGD, so the notion of similar dialogues follows the task they
serve.

## Citation

Fernández, J. M., Errecalde, M., & Burdisso, S. *TRACE-Dialog: Trajectory-Aware Dialogue Embeddings from Contextual
Turn Representations.* In preparation.
