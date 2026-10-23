---
license: cc-by-nc-4.0
language: en
base_model: Qwen/Qwen3.5-9B
tags:
  - dialog2agent
  - task-oriented-dialogue
  - spokenwoz
  - lora
---

# dialog2agent: SpokenWOZ domain artifact

Domain artifact of [dialog2agent](https://github.com/jumafernandez/dialog2agent) for SpokenWOZ: everything that the
agent learned from the 4,200 training dialogues of the corpus, without using their annotations. With it, the library
builds an agent that locates the conversation in an induced state, retrieves similar situations of other dialogues,
decides the next action by scoring the actions that the structure allows in that state, and writes the response.

[Documentation](https://jumafernandez.github.io/dialog2agent/) ·
[Web demo](https://jumafernandez.github.io/dialog2agent/demo.html)

## Usage

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("jumafernandez/dialog2agent-spokenwoz")
session = agent.new_session()
result = agent.turn(session, "hi, i'm looking for a cheap restaurant in the centre of town")
print(result.state, result.action.name, result.options[0].probability)
print(result.text)
```

The library downloads the base model and the encoders from the Hub at the revisions pinned in `domain.json`. The
agent needs an accelerator with about 24 GB of memory, such as an NVIDIA L4.

## Contents

| Path | Content |
|---|---|
| `domain.json` | base model, encoders with their revisions, settings of the agent, scope and attribution |
| `structure/` | 68 system actions, 73 user actions and 146 states induced with θ = 0.66 and τ = 0.01, with the counts of the transitions between each state and the next system action |
| `memory/` | 74,524 situations of the training dialogues, represented by `[e_t ; 0.75 h_CLS]`, with the user turn, the next action, the response and the dialogue of origin |
| `adapter/` | LoRA adapter of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at commit `c202236`, trained on the same dialogues without their annotations |

The turns are represented with [Dialog2Flow](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base),
[TRACE](https://huggingface.co/jumafernandez/trace-deep-ar) and
[TRACE-Dialog](https://huggingface.co/jumafernandez/trace-dialog-deep-tod).

## Evaluation

On the 17,782 system turns of the 1,000 test dialogues of SpokenWOZ, with the real history of each dialogue, the agent
built from this artifact decided the action of the operator in 55.6% of the turns, and its responses obtained an
Inform of 70.6, a Success of 53.7, a BLEU of 23.2 and a Comb of 85.4 with the official evaluator of the corpus. The
[documentation](https://jumafernandez.github.io/dialog2agent/evaluation/) describes this evaluation and its
comparison with the evaluation of the thesis.

## Scope

`domain.json` declares the scope of the domain: a user turn is out of scope when its Dialog2Flow embedding is far from
every induced user action or when the memory holds no similar situation. The thresholds are the 1st percentile of each
signal over the 9,169 user turns of the development dialogues, which were not used to build the artifact. The agent
was built from dialogues in English, so turns in other languages usually fall out of scope.

## License and attribution

The artifact contains data derived from SpokenWOZ, which is distributed under the Creative Commons
Attribution-NonCommercial 4.0 International license, so the artifact is distributed under the same license and may be
used for non-commercial purposes only, with attribution:

> Si, S., Ma, W., Gao, H., Wu, Y., Lin, T.-E., Dai, Y., Li, H., Yan, R., Huang, F. and Li, Y. (2023). SpokenWOZ: A
> Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents. NeurIPS 2023 Datasets and Benchmarks
> Track. https://spokenwoz.github.io/SpokenWOZ-github.io/

The library and the TRACE encoders are distributed under the Apache License 2.0, and the base model under its own
license.
