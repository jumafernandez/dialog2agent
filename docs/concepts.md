# Concepts

## Representations

- **Turn embeddings** (Dialog2Flow) characterize the action each turn performs.
- **Contextual representations** (TRACE) incorporate the history of the dialogue into each turn, using only past
  turns, and locate the conversation in a state.
- **Dialogue representations** (TRACE-Dialog) summarize the dialogue up to the current turn and are used to retrieve
  similar situations.

## Induced structure

Actions are regions of the Dialog2Flow space, and states are regions of the combination of the turn embeddings with
the contextual representation. For each state, the allowed actions are the ones that followed it in the dialogues with
a probability above a threshold. Inducing the structure requires no annotations.

## Conversational memory

Each entry is a user turn of a stored dialogue, with the action the system took next and its response. The memory
retrieves the situations most similar to the current one, optionally restricted to the same state or to allowed
actions.

## Deciding by scoring

Instead of generating a free description of the next action, the language model scores every allowed action and the
agent chooses the best one. The decision is a choice among options defined by the structure, so it never leaves the
allowed set and it returns a probability for each option.

## Acting and writing

When the decided action needs information from the environment, a tool is called, such as a database search. The
response is written after the decided action, with the tool results in the prompt.
