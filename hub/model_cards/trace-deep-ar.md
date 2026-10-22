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
  - contextual-representations
  - turn-embeddings
---

# TRACE (deep, autoregressive)

TRACE is a contextual turn encoder for task-oriented dialogue. A frozen base encoder maps each turn to a static
embedding $e_t$, and TRACE, a Transformer whose tokens are turns, maps the sequence $(e_1, \dots, e_T)$ to contextual
representations $h_t$. In the autoregressive mode each turn attends only to the turns before it, so $h_t$ can be
computed while the conversation is ongoing. Training is self-supervised and uses no dialogue act annotations.

## Model details

| | |
|---|---|
| Base turn encoder (frozen) | [`sergioburdisso/dialog2flow-joint-bert-base`](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base) |
| Architecture | BERT-style encoder over turn embeddings: 12 layers, 12 heads, hidden size 768 |
| Parameters | 85.1M |
| Attention | autoregressive (causal over turns) |
| Inputs | turn embeddings (768), speaker ids and turn positions, up to 64 turns per window |
| Output | $h_t \in \mathbb{R}^{768}$ for every turn |
| Training data | task-oriented dialogue collection of Dialog2Flow, partitioned by near-duplicate clusters |
| Training | from scratch, self-supervised next-turn prediction, learning rate $10^{-4}$, seed 42, checkpoint selected by validation loss |

## Usage

The model ships its own code, so it loads with `trust_remote_code=True`. Pinning a `revision` is recommended.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("jumafernandez/trace-deep-ar", trust_remote_code=True)

turns = ["hi, I need a cheap restaurant in the centre", "sure, what kind of food would you like?", "italian"]
speakers = ["user", "system", "user"]
h = model.encode_dialogue(turns, speakers)   # (3, 768), one h_t per turn
```

`encode_dialogue` computes the base embeddings with the base encoder declared in the configuration, which requires `sentence-transformers`. Precomputed base
embeddings can be passed with `embeddings=`. Dialogues longer than 64 turns are encoded in consecutive windows.

## Evaluation

In the retrieval of conversational situations on SpokenWOZ, where the next dialogue act is predicted by the vote of
the continuations of the $k$ nearest turns (1,019 single-domain training dialogues, cross-validated), $h_t$ alone
performs at the level of the base embedding, while the combination $[e_t ; \lambda h_t]$ improves on it:

| Representation | Accuracy | Macro-F1 | Coverage@5 |
|---|---|---|---|
| $e_t$ (Dialog2Flow) | 0.454 | 0.135 | 0.646 |
| $h_t$ (this model) | 0.453 | 0.143 | 0.664 |
| $[e_t ; \lambda h_t]$ | 0.483 | 0.158 | 0.686 |

As a component of the [dialog2agent](https://github.com/jumafernandez/dialog2agent) agent, which locates the
conversation in a state of an induced structure, adding $h_t$ to $e_t$ raises the accuracy of a pointer decider of
the next system action on the SpokenWOZ test set from 0.436 to 0.471.

This repository contains the checkpoint converted to the transformers format. The conversion was verified against
the representations computed with the training code on 50 SpokenWOZ dialogues, with a maximum absolute difference of
$1.2 \times 10^{-5}$.

## Limitations

The model was trained on English task-oriented dialogue and expects embeddings of the base encoder it was trained
on. Its representations are useful as a complement of the base turn embeddings rather than as a replacement.

## Citation

Fernández, J. M., Errecalde, M., & Burdisso, S. *TRACE: Learning the Update for Contextual Turn Representations in
Task-Oriented Dialogue.* In preparation.
