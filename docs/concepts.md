# Concepts

## Representations

- **Turn embeddings** (Dialog2Flow, `e_t`) characterize the action that each turn performs.
- **Contextual representations** (TRACE, `h_t`) incorporate the history of the dialogue into each turn, using only
  past turns, and locate the conversation in a state.
- **Dialogue representations** (TRACE-Dialog, `h_CLS`) summarize the dialogue up to the current turn and are used to
  retrieve similar situations.

## Induced structure

Actions are regions of the Dialog2Flow space, and states are regions of the combination `[e_t ; λ h_t]` of the turn
embeddings with the contextual representations. For each state, the allowed actions are the actions that followed it
in the dialogues with a probability above a threshold. Inducing the structure requires no annotations, and the
actions are named from the turns that they group.

## Conversational memory

Each entry of the memory is a user turn of a stored dialogue, represented by `[e_t ; λ h_CLS]`, together with the
action that the system took next, its response, the dialogue it comes from and how that dialogue ended. The memory
retrieves the situations most similar to the current one, optionally restricted to the same state or to situations
whose next action is allowed.

## Deciding by scoring

Instead of generating a free description of the next action, the language model scores every allowed action, that
is, the log-probability of the output `Action: <name>` followed by `Response:` after the prompt, and the agent chooses
the most probable one. The decision is a choice among options defined by the structure, so it never leaves the
allowed set and it returns a probability for each option, in the line of decision models such as
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). Regions that share a name receive the same
score, and the region that is more probable in the state is chosen.

The agent can also decide by generation (`decision="generate"`), as the reference of the evaluation: the model writes
the action and the response, and the declared action is projected to the closest induced action.

## Acting and writing

When the decided action needs information from the environment, a tool is called, such as a database search. The
response is written after the decided action, with the tool results in the prompt.

## Scope

A domain artifact can declare its scope, which tells whether a user turn resembles the turns of the dialogues from
which the agent was built. The check uses the structure and the memory themselves: a turn is out of scope when its
Dialog2Flow embedding is far from every induced user action or when the memory holds no similar situation (see
[Build your own domain](build-a-domain.md#scope)).
