# Getting started

## Installation

```bash
pip install dialog2agent
```

## Talk to an agent

```bash
dialog2agent chat domains/spokenwoz
```

## Use it from Python

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("domains/spokenwoz")
session = agent.new_session()
for text in ["hello, I need a restaurant in the centre", "something cheap, please"]:
    result = agent.turn(session, text)
    print(f"[{result.action.name}] {result.text}")
```
