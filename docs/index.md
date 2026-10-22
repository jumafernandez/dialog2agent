# dialog2agent

**Build task-oriented dialogue agents that decide over structures induced from your own conversations.**

`dialog2agent` turns a collection of unannotated dialogues into an agent. It induces a structure of states and
actions from the dialogues, builds a memory of conversational situations and adapts a language model. At each turn,
the agent locates the conversation in a state, retrieves similar situations from other dialogues, decides the next
action by scoring the actions the structure allows in that state, and writes the response.

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("domains/spokenwoz")
session = agent.new_session()
result = agent.turn(session, "I'm looking for a cheap hotel in the north")
print(result.action.name, result.options[0].probability)
print(result.text)
```

Every turn returns the full trace of the decision: the state, the allowed actions with their probabilities, the
retrieved situations and the tool calls.
