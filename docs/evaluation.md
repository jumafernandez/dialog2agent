# Evaluation

The agent of the thesis was evaluated on SpokenWOZ at three levels: the action (accuracy of the decided action and
coherence with the allowed actions), the human annotation (agreement with the annotated dialogue acts, which the agent
never uses) and the task (Inform, Success and BLEU with the official evaluator of the corpus). The library predicts the
turns of a corpus and computes the metrics of the action, and its predictions contain the responses from which the
official evaluator computes the metrics of the task. A second evaluation puts the agent in complete conversations
with simulated users.

## Predicting a corpus

`dialog2agent evaluate` predicts every system turn that follows a user turn with `Agent.respond`, from the real
dialogue up to that user turn, so that each turn is evaluated with the same history as the reference. The dialogues
are JSON lines:

```json
{"dialogue_id": "MUL0001", "turns": [{"speaker": "user", "text": "..."}, {"speaker": "system", "text": "..."}]}
```

```bash
dialog2agent evaluate my-domain dialogues.jsonl --output predictions/
```

Each prediction records the state, the allowed actions, the decided action and its probability, the response, and
the induced action of the real system turn, that is, the system action of the structure closest to its Dialog2Flow
embedding. The evaluation therefore needs no annotations.

With `--no-write` only the actions are decided, which is much faster. Runs can be interrupted and resumed, since
turns already present in the output are skipped.

## Several accelerators

Each process takes the dialogues of its rank and writes its own file, so the evaluation runs with
`torch.distributed.run` without communication between processes:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 8 -m dialog2agent.cli evaluate \
    my-domain dialogues.jsonl --output predictions/ --require-accelerator
```

On a cluster without access to the Hub, `--base-model` and `--turn-encoder` point to local copies of the models, and
the TRACE encoders are read from the Hugging Face cache at their pinned revisions.

## Metrics

```bash
dialog2agent report predictions/
```

reports the accuracy of the decided action by region and by name, and the coverage of the real action by the allowed
actions, over all the predicted turns. Turns whose real action is not allowed count as errors.

## Reproduction of the thesis

The evaluation of the thesis was repeated with the library on the 17,782 system turns of the 1,000 test dialogues of
SpokenWOZ. The state, the allowed actions and the induced action of the reference matched the thesis in every turn,
and the decided action matched in 96.6% of them. The remaining differences come from two implementation details that
only affect ties between equivalent options: the thesis computed the similarities of the memory in bfloat16, and it
broke ties between regions with the same name and count by their order of appearance.

| | Accuracy (region) | Accuracy (name) | Inform | Success | BLEU | Comb |
|---|---|---|---|---|---|---|
| Thesis | 0.557 | 0.570 | 70.9 | 54.0 | 23.2 | 85.6 |
| dialog2agent | 0.556 | 0.569 | 70.6 | 53.7 | 23.2 | 85.4 |

The difference in accuracy, -0.0005 (95% CI from -0.0024 to 0.0014, paired bootstrap by dialogue), is not
significant.

## Closed-loop simulation with SDialog

The evaluation above predicts each turn from the real history, so a wrong decision does not change the following
turns. To observe the agent in complete conversations, simulated users of [SDialog](https://github.com/idiap/sdialog)
talk to the agent through the OpenAI-compatible API of the web service (see [Serving the agent](serving.md)). Each
user receives a persona whose background states a goal of a test dialogue: the constraints of each service, the
information to request, the booking and the personal details.

```python
from sdialog.agents import Agent
from sdialog.personas import Persona

persona = Persona(name="Alex Smith", language="English",
                  role="a customer who calls the Cambridge town information centre on the phone",
                  background="You are looking for a train from ely to cambridge on wednesday that arrives by 08:15. "
                             "You want to know its departure time.")
user = Agent(persona=persona, name="User", model="openai:user-model", openai_api_base="http://localhost:8101/v1",
             openai_api_key="EMPTY", temperature=0)
system = Agent(name="System", model="openai:dialog2agent", openai_api_base="http://localhost:8000/v1",
               openai_api_key="EMPTY", temperature=0, can_finish=False)
dialog = user.dialog_with(system, max_turns=40)
dialog.to_file("dialog.json")
```

In the thesis, the simulated user was the 27 billion parameter model of the Qwen 3.5 family, the conversations were
judged by Claude Opus 5.5 without knowing which variant produced them, and the differences between variants were
estimated with a paired bootstrap by goal. The [web demo](https://jumafernandez.github.io/dialog2agent/demo.html)
replays some of these conversations with the trace of every decision.
