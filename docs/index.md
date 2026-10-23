# dialog2agent

**Build task-oriented dialogue agents that decide over structures induced from your own conversations.**

[Try the web demo](https://jumafernandez.github.io/dialog2agent/demo.html){ .md-button .md-button--primary }
[Get started](getting-started.md){ .md-button }

`dialog2agent` turns a collection of unannotated dialogues into an agent. It induces a structure of states and
actions from the dialogues, builds a memory of conversational situations and adapts a language model. At each turn,
the agent locates the conversation in a state, retrieves similar situations from other dialogues, decides the next
action by scoring the actions that the structure allows in that state, and writes the response.

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("jumafernandez/dialog2agent-spokenwoz")
session = agent.new_session()
result = agent.turn(session, "I'm looking for a cheap hotel in the north")
print(result.state, result.action.name, result.options[0].probability)
print(result.text)
```

Every turn returns the full trace of the decision, that is, the state, the allowed actions with their probabilities,
the retrieved situations and the tool calls. Therefore, each response can be attributed to a state and an action of
the structure.

## Components

| Component | Role |
|---|---|
| Turn representations ([Dialog2Flow](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base)) | characterize the action that each turn performs |
| Contextual representations ([TRACE](models.md)) | locate the conversation in a state |
| Dialogue representations ([TRACE-Dialog](models.md)) | retrieve similar situations of other dialogues |
| Induced structure | defines the actions, the states and the actions allowed in each state |
| Conversational memory | situations of other dialogues, with what the system did next |
| Language model | decides among the allowed actions and writes the response |
| Tools | act on the environment, such as searching a database |

## Background

The library implements the hybrid agent of the doctoral thesis of Juan Manuel Fernández at the Universidad Nacional de
San Luis (Argentina). The agent was evaluated on SpokenWOZ without using the annotations of the corpus, and the
library reproduces that evaluation (see [Evaluation](evaluation.md)).

## License

The library and the TRACE encoders are distributed under the Apache License 2.0. Domain artifacts, and any data
derived from a corpus, keep the license of that corpus: the SpokenWOZ artifact and the data of the web demo are
distributed under CC BY-NC 4.0, for non-commercial use with attribution.
