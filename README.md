# dialog2agent

**Build task-oriented dialogue agents that decide over structures induced from your own conversations.**

`dialog2agent` turns a collection of unannotated dialogues into an agent. From the dialogues it induces a structure
of states and actions, builds a memory of conversational situations and adapts a language model. At each turn, the
agent locates the conversation in a state, retrieves similar situations from other dialogues, decides the next action
by scoring the actions the structure allows in that state, and writes the response.

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("domains/spokenwoz")
session = agent.new_session()
result = agent.turn(session, "I'm looking for a cheap hotel in the north")

result.text         # the response
result.action       # the decided action
result.options      # allowed actions with their probabilities
result.situations   # retrieved situations from other dialogues
```

Every turn returns the full trace of the decision, which makes the agent observable and each response attributable to
a state and an action of the structure.

## How it works

| Component | Role |
|---|---|
| Turn representations (Dialog2Flow) | characterize the action each turn performs |
| Contextual representations (TRACE) | locate the conversation in a state |
| Dialogue representations (TRACE-Dialog) | retrieve similar situations of other dialogues |
| Induced structure | defines the actions, the states and the actions allowed in each state |
| Conversational memory | situations of other dialogues with what the system did next |
| Language model | decides among the allowed actions and writes the response |
| Tools | act on the environment, such as searching a database |

## Status

Under active development. The first milestone ports the agent evaluated on SpokenWOZ, which decides without using
the annotations of that corpus.

## Installation

```bash
pip install dialog2agent            # inference
pip install "dialog2agent[build]"   # build a domain from your dialogues
pip install "dialog2agent[eval]"    # evaluation
```
