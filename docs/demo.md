# Web demo

[Open the web demo](https://jumafernandez.github.io/dialog2agent/demo.html){ .md-button .md-button--primary }

The demo shows the agent at work. A conversation runs next to the induced structure: the states are drawn on the
plane of the first two principal components of their centers, the current state is highlighted and the path of the
conversation is traced over them. For each turn of the agent, the decision panel shows the allowed actions of the
state with the probability that the model assigns to each one, and the memory panel shows the situations of other
conversations that the agent retrieved. Hovering over a state, or tapping it, shows its most probable actions and two
typical user turns of that state.

## How the agent decides

A selector at the top switches between three ways of deciding the action, with their reference metrics. The agent
decides as the decision models of [TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev), such
as Jev: instead of writing the action, it scores every option of a set defined in advance, here the actions allowed in
the state, and returns a probability for each one, with the same adapted model and open weights. The other two
variants are the same adapted model generating the action and the base model without the adapter. The replay shows
the same goal solved by each of them.

## Replay and live conversation

By default the demo replays conversations of the closed-loop simulation with
[SDialog](https://github.com/idiap/sdialog), in which simulated users with goals of the SpokenWOZ test set talk to the
three variants (see [Evaluation](evaluation.md#closed-loop-simulation-with-sdialog)). The goals are selected by
domain, without looking at the judgments, and the verdict of the judge is shown for each conversation.

The button "talk to the agent" wakes up the agent served on Modal (see [Serving the agent](serving.md)) and opens a
live conversation with the first two variants, which share the adapted model. The page never wakes the service on its
own, since every request to a stopped service starts a GPU; the first start takes about a minute. In the live
conversation the agent has tools: when the action it decides is a tool action, it searches the SpokenWOZ database or
makes a simulated booking, and the call, with its arguments and its result, is shown under the response. The replay
comes from a simulation in which the agent had no tools yet, so the turns in which it announces a search get a note.

Before a turn of the visitor reaches the agent, the page checks that it is in English, by the function words of each
language, and the service checks that it is within the scope of the domain (see
[Build your own domain](build-a-domain.md#scope)). The domain is checked until the first turn within it, that is,
on the request that opens the conversation, since closings and short turns such as spelled names often fall below the
thresholds. A turn out of scope stays in the conversation, marked, with a notice that shows the value of each signal
and its threshold, and it is not sent to the agent.

## Build and serve

The replay data is built from a simulation directory and the domain artifact:

```bash
python demo/build_replay.py --domain DOMAIN_DIR --simulation SIM_DIR --out demo/data/replay.json
python -m http.server --directory demo 8000    # then open http://localhost:8000/demo.html
```

The replay data contains text derived from the dialogues of SpokenWOZ, so it is distributed under CC BY-NC 4.0, with
the attribution that the page shows in its footer (see `demo/data/LICENSE.md`). The site of this documentation
publishes the demo at `demo.html`, with the workflow `.github/workflows/pages.yml`.
