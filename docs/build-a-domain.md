# Build your own domain

A domain artifact contains everything the agent learns from a collection of dialogues: the induced structure, the
memory, the adapter of the language model and, optionally, the database of the tools.

```bash
dialog2agent build my-dialogues.jsonl --output domains/my-domain
```

Each line of the input file is a dialogue with its turns:

```json
{"dialogue_id": "d1", "turns": [{"speaker": "user", "text": "hi"}, {"speaker": "system", "text": "hello"}]}
```

!!! note
    The `build` command is under development.
