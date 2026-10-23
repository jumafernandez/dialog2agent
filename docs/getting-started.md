# Getting started

## Installation

The library requires Python 3.10 or later and is installed from its repository:

```bash
pip install "dialog2agent @ git+https://github.com/jumafernandez/dialog2agent"
```

Optional dependencies are installed with extras: `web` to serve the agent, `eval` for the evaluation, `build` to build
a domain and `docs` for this documentation.

```bash
pip install "dialog2agent[web] @ git+https://github.com/jumafernandez/dialog2agent"
```

## Domain artifacts

An agent is loaded from a domain artifact, a directory with everything that the agent learned from the dialogues of
a domain: the induced structure, the memory of situations, the adapter of the language model and the settings in
`domain.json` (see [Build your own domain](build-a-domain.md)). The language model and the encoders are downloaded
from the Hugging Face Hub the first time they are used, at the revisions pinned by the artifact and by the library.

The SpokenWOZ artifact is published on the Hub as
[`jumafernandez/dialog2agent-spokenwoz`](https://huggingface.co/jumafernandez/dialog2agent-spokenwoz), under CC BY-NC
4.0 since it contains data derived from SpokenWOZ, and `Agent.from_pretrained` and the command line accept either a
local directory or a repository of the Hub. The agent can also be tried without installing anything in the
[web demo](https://jumafernandez.github.io/dialog2agent/demo.html).

## Talk to an agent

```bash
dialog2agent chat jumafernandez/dialog2agent-spokenwoz
```

`--device` selects the accelerator (for example `cuda`, `mps`, `xpu` or `cpu`), `--decision generate` makes the
model generate the action instead of scoring the allowed actions, and `--no-adapter` loads the base model without the
adapter of the artifact.

## Use it from Python

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("jumafernandez/dialog2agent-spokenwoz")
session = agent.new_session()
for text in ["hello, I need a restaurant in the centre", "something cheap, please"]:
    result = agent.turn(session, text)
    print(f"[state {result.state} · {result.action.name}] {result.text}")
```

`Agent.turn` adds the user turn and the response to the session. `Agent.respond` computes the next system turn of any
dialogue that ends with a user turn without changing any state, and it is the method that the evaluation calls, so
the agent that is used is the agent that was evaluated.

Every result carries the trace of the decision:

| Field | Content |
|---|---|
| `text` | the response |
| `action` | the decided action, with its `id` and `name` |
| `state` | the induced state of the conversation |
| `options` | the allowed actions with their probability, from the most to the least probable |
| `situations` | the situations retrieved from the memory, with their similarity |
| `tool_calls` | the calls to tools triggered by the action |

## Hardware

The SpokenWOZ agent runs the 9 billion parameter model of the Qwen 3.5 family in bfloat16, so it needs an accelerator
with about 24 GB of memory, such as an NVIDIA L4. The encoders are small and also run on CPU.
