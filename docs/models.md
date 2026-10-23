# Models

The agent uses three encoders. [Dialog2Flow](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base)
(Burdisso et al., 2024) represents each turn by the action that it performs. TRACE and TRACE-Dialog, trained in the
doctoral thesis on which the library is based, read the Dialog2Flow embeddings of the turns and are published on the
Hugging Face Hub under the Apache License 2.0.

| Model | Repository | Output | Use in the agent |
|---|---|---|---|
| TRACE, deep, autoregressive | [`jumafernandez/trace-deep-ar`](https://huggingface.co/jumafernandez/trace-deep-ar) | one vector `h_t` per turn, from that turn and the previous ones | locating the state |
| TRACE-Dialog, deep, with the similarity stage | [`jumafernandez/trace-dialog-deep-tod`](https://huggingface.co/jumafernandez/trace-dialog-deep-tod) | one vector `h_CLS` per dialogue | querying the memory |

Both models ship their own code, so they load with `trust_remote_code=True`, and they compute the Dialog2Flow
embeddings of the turns with the base encoder declared in their configuration.

## TRACE with transformers

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("jumafernandez/trace-deep-ar", trust_remote_code=True)

turns = ["hi, I need a cheap restaurant in the centre", "sure, what kind of food would you like?", "italian"]
speakers = ["user", "system", "user"]
h = model.encode_dialogue(turns, speakers)   # (3, 768), one h_t per turn
```

## TRACE-Dialog with sentence-transformers

Each input is a whole dialogue, given as a list of chat messages (`role` and `content`) or of turns (`speaker` and
`text`).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jumafernandez/trace-dialog-deep-tod", trust_remote_code=True)

dialogue = [{"role": "user", "content": "hi, I need a cheap restaurant in the centre"},
            {"role": "assistant", "content": "sure, what kind of food would you like?"},
            {"role": "user", "content": "italian"}]
embeddings = model.encode([dialogue])   # (1, 768), one h_CLS per dialogue
```

The embeddings can be compared with the similarity functions of sentence-transformers, for example to retrieve the
dialogues of a collection that are closest to an ongoing one. TRACE-Dialog also loads with `AutoModel`, whose
`encode_dialogue(turns, speakers)` returns `h_CLS`.

## Within the library

`TraceEncoder` and `TraceDialogEncoder` load the models at the revisions pinned in
`dialog2agent.representations.REVISIONS`, so that a later update of the repositories does not change the behavior of
an agent. A domain artifact can pin other revisions in `domain.json`.

## Verification

The published models were compared with the models used in the thesis on dialogues of SpokenWOZ:

| Model | Format | Comparison | Max. absolute difference |
|---|---|---|---|
| TRACE | transformers | `h_t`, 50 dialogues | 1.2e-5 |
| TRACE-Dialog | transformers | `h_CLS`, 1,882 prefixes | 7.4e-6 |
| TRACE-Dialog | sentence-transformers | `h_CLS` from the text, 500 prefixes | 1.4e-5 |
