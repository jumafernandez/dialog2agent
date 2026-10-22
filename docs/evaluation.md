# Evaluation

The agent is evaluated on SpokenWOZ at three levels: the action (accuracy of the decided action and coherence with
the allowed actions), the human annotation (agreement with the annotated dialogue acts, which the agent never uses) and
the task (Inform, Success and BLEU with the official evaluator of the corpus).

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
