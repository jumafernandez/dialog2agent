"""Evaluation of an agent on a corpus of dialogues.

Each system turn that follows a user turn is predicted with ``Agent.respond`` from the real dialogue up to that user
turn, so that every turn is evaluated with the same history as the reference. The decided action is compared with the
induced action of the real system turn, that is, the system action of the structure closest to its Dialog2Flow
embedding, so the evaluation needs no annotations. Predictions are written as JSON lines, one file per process, and
the metrics are computed from those files.

Format of the dialogues (JSON lines)::

    {"dialogue_id": "...", "turns": [{"speaker": "user", "text": "..."}, {"speaker": "system", "text": "..."}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

from .types import Turn


def load_dialogues(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def system_turns(turns: list[Turn]) -> list[int]:
    """Positions of the system turns that follow a user turn."""
    return [i for i in range(1, len(turns)) if turns[i].speaker == "system" and turns[i - 1].speaker == "user"]


def _done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {(r["dialogue_id"], r["turn_id"]) for r in map(json.loads, f)}


def predict(agent, dialogues: list[dict], output: str | Path, write: bool = True, shard: int = 0,
            num_shards: int = 1, log_every: int = 0) -> int:
    """Predict the system turns of the dialogues of one shard (``dialogues[shard::num_shards]``) and append them to
    ``output``. Turns already present in ``output`` are skipped, so an interrupted run can be resumed. Returns the
    number of new predictions."""
    output = Path(output)
    done, names, n = _done(output), agent.structure.names, 0
    with open(output, "a") as f:
        for d in dialogues[shard::num_shards]:
            turns = [Turn(t["speaker"], t.get("text") or "") for t in d["turns"]]
            for i in system_turns(turns):
                if (d["dialogue_id"], i) in done:
                    continue
                r = agent.respond(turns[:i], write=write)
                ref = int(agent.structure.system_action(agent.turn_encoder.encode([turns[i].text]))[0])
                f.write(json.dumps({
                    "dialogue_id": d["dialogue_id"], "turn_id": i, "state": r.state,
                    "allowed": agent.structure.allowed(r.state), "action": r.action.id, "action_name": r.action.name,
                    "probability": r.options[0].probability, "reference_action": ref,
                    "reference_action_name": names[ref], "response": r.text, "reference": turns[i].text},
                    ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if log_every and n % log_every == 0:
                    print(f"  {n} turns", flush=True)
    return n


def report(paths: list[str | Path]) -> dict:
    """Accuracy of the decided action by region and by name, and coverage of the real action by the allowed ones,
    over all the predicted turns (turns whose real action is not allowed count as errors)."""
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f]
    if not rows:
        return {"turns": 0}
    n = len(rows)
    return {"turns": n, "dialogues": len({r["dialogue_id"] for r in rows}),
            "action_accuracy": sum(r["action"] == r["reference_action"] for r in rows) / n,
            "action_name_accuracy": sum(r["action_name"] == r["reference_action_name"] for r in rows) / n,
            "coverage": sum(r["reference_action"] in r["allowed"] for r in rows) / n}
