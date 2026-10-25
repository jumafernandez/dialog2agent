# Build your own domain

A domain artifact contains everything that the agent learns from a collection of dialogues, so the same agent code
serves any domain for which an artifact is available.

```
my-domain/
  domain.json        # base model, encoders and agent settings
  structure/         # induced structure (structure.npz, structure.json)
  memory/            # situations (vectors.npy, entries.jsonl, memory.json)
  adapter/           # LoRA adapter of the language model (optional)
  db/                # database of the tools (optional, <domain>_db.json and booking.json)
```

## domain.json

| Key | Content |
|---|---|
| `base_model` | language model, for example `Qwen/Qwen3.5-9B` |
| `turn_encoder` | Dialog2Flow turn encoder |
| `context_encoder`, `context_encoder_revision` | TRACE encoder and its commit |
| `dialogue_encoder`, `dialogue_encoder_revision` | TRACE-Dialog encoder and its commit |
| `k`, `conditioned` | number of retrieved situations and whether retrieval is conditioned by the structure |
| `decision_scale` | calibration of the probabilities of the decision |
| `max_tokens` | budget of the prompt |
| `search_triggers` | keywords of the action names that call the search tool |
| `tools` | induced actions, by id, that call each tool, for example `{"search": [43, 9], "book": [8]}` |
| `tool_results` | `"message"` to give the tool results to the model as a message of the tool role |
| `scope` | thresholds of the scope of the domain (see below) |
| `data` | name, citation, URL and license of the dialogues from which the artifact was built |

`db/booking.json` declares the parameters of the booking of each domain, for example
`{"restaurant": ["people", "day", "time"]}`, which belong to the booking interface and not to the database. A file of
the database that holds a single object of value lists, such as the taxis of MultiWOZ, describes a service without
entities, whose bookings are assembled from those lists.

The revisions of the encoders are pinned because the memory is computed with a given version of each encoder. The
`data` entry is returned by the web service, so that any interface built on it can show the attribution.

## Building

```bash
dialog2agent build my-dialogues.jsonl --output domains/my-domain
```

Each line of the input file is a dialogue with its turns:

```json
{"dialogue_id": "d1", "turns": [{"speaker": "user", "text": "hi"}, {"speaker": "system", "text": "hello"}]}
```

!!! note
    The `build` command is under development. The SpokenWOZ artifact was exported from the structure, the memory and
    the adapter of the thesis.

## Scope

The scope of a domain is calibrated on user turns that were not used to build the artifact, such as those of its
development dialogues. Each threshold is a low percentile of its signal over those turns, and a turn is out of scope
when either signal falls below its threshold.

```python
from dialog2agent import Agent, Scope

agent = Agent.from_pretrained("domains/my-domain")
scope = Scope.calibrate(agent, development_user_turns, percentile=1, language="en",
                        description="calls to a town information centre")
print(scope.to_dict())   # stored under "scope" in domain.json
print(agent.check_scope("what is the weather like tomorrow?"))
```

For SpokenWOZ, with the 1st percentile over the 9,169 user turns of the development dialogues, 1.7% of 5,000 user
turns of the test set fall out of scope, mostly closings and turns with transcription errors. All the messages in
Spanish written for the check fall out of scope as well, since the encoders were trained on dialogues in English,
and so do 16 of 30 requests in English unrelated to the domain.
