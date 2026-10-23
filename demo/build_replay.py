"""Build the replay data of the web demo from a closed-loop simulation and the domain artifact.

The simulation directory is the output of the SDialog simulation of the thesis: one directory per agent variant with
one SDialog dialogue per goal, the ``metas.json`` file with the goal of each simulated user, the trace logs written by
``dialog2agent serve --log`` (``trazas_<variant>.jsonl``) and, optionally, ``evaluacion_por_dialogo.jsonl`` with the
judgments. For each selected goal and each variant, every system turn is matched with the trace of the request that
produced it, so the demo shows the state, the allowed actions, the decision and the retrieved situations of each turn.
Each state also carries its most probable allowed actions and two typical user turns of the memory of the domain (the
most frequent ones of moderate length), which the demo shows when the pointer passes over the state.

Selection: for each domain, the first goals (by identifier) that the three variants simulated and whose dialogue with
the reference variant has a number of turns in the given range. The judgments do not intervene in the selection; they
are shown as they are. The states are placed on the plane by the first two principal components of their centers.

Usage:
    python demo/build_replay.py --domain DOMAIN_DIR --simulation SIM_DIR --out demo/data/replay.json \
        [--per-domain 2] [--turns 6 16]

The output contains text derived from the dialogues of the domain, so it is distributed under the license of those
dialogues (declared in ``domain.json``), not under the license of the library.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ["puntuacion", "generacion", "sin_adaptar"]
STOP_TEXT = "(bye bye!)"


def norm(text: str) -> str:
    return " ".join(text.replace(STOP_TEXT, "").split())


def typical_turns(memory_dir: Path, n_states: int, k: int = 2) -> list[list[str]]:
    """The ``k`` most frequent user turns of each state among the situations of the memory (12 to 90 characters)."""
    counts = [Counter() for _ in range(n_states)]
    path = memory_dir / "entries.jsonl"
    if path.exists():
        for line in path.open():
            e = json.loads(line)
            text = " ".join(e["user"].split())
            if e.get("state") is not None and 12 <= len(text) <= 90:
                counts[e["state"]][text] += 1
    return [[t for t, _ in c.most_common(k)] for c in counts]


def load_structure(structure_dir: Path, memory_dir: Path):
    arrays = np.load(structure_dir / "structure.npz")
    meta = json.loads((structure_dir / "structure.json").read_text())
    centers, transitions = arrays["state_centers"].astype(np.float64), arrays["transitions"].astype(np.float64)
    centered = centers - centers.mean(0)
    plane = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
    plane = (plane - plane.min(0)) / (plane.max(0) - plane.min(0))
    examples = typical_turns(memory_dir, len(centers))
    states, allowed = [], []
    for i, row in enumerate(transitions):
        counts = row if row.sum() else transitions.sum(0)
        prior = counts / counts.sum()
        order = np.argsort(-prior, kind="stable")
        allowed.append([int(a) for a in order if prior[a] >= meta["tau"]] or [int(order[0])])
        states.append({"x": round(float(plane[i, 0]), 4), "y": round(float(plane[i, 1]), 4),
                       "n": int(row.sum()), "examples": examples[i],
                       "allowed": [[a, round(float(prior[a]), 3)] for a in allowed[-1][:5]]})
    return {"states": states, "actions": meta["names"]}, allowed


def steps(turns: list[dict], traces: dict, allowed: list[list[int]]) -> list[dict]:
    """Trace of every system turn: the request whose user turns and response match the dialogue."""
    out = []
    for k, turn in enumerate(turns):
        if turn["speaker"] != "System":
            continue
        key = (tuple(norm(t["text"]) for t in turns[:k] if t["speaker"] == "User"), norm(turn["text"]))
        if key not in traces:
            raise ValueError(f"no trace for turn {k}")
        r = traces[key]
        out.append({"turn": k, "state": r["state"], "action": r["action"]["id"], "allowed": allowed[r["state"]],
                    "options": [{"id": o["id"], "p": round(o["probability"], 4)} for o in r["options"][:6]],
                    "situations": [{"user": s["user"], "action": s["action"], "response": s["response"],
                                    "similarity": round(s["similarity"], 4)} for s in r["situations"][:3]]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=Path, required=True)
    ap.add_argument("--simulation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-domain", type=int, default=2)
    ap.add_argument("--turns", type=int, nargs=2, default=[6, 16])
    a = ap.parse_args()
    structure, allowed = load_structure(a.domain / "structure", a.domain / "memory")
    goals = {g["dialogue_id"]: g for g in json.loads((a.simulation / "metas.json").read_text())}
    evaluation = a.simulation / "evaluacion_por_dialogo.jsonl"
    judged = ({(r["variante"], r["dialogue_id"]): r for r in map(json.loads, evaluation.open())}
              if evaluation.exists() else {})
    traces = {}
    for v in VARIANTS:
        traces[v] = {}
        for line in (a.simulation / f"trazas_{v}.jsonl").open():
            r = json.loads(line)
            key = (tuple(norm(t["text"]) for t in r["turns"] if t["speaker"] == "user"), norm(r["result"]["text"]))
            traces[v][key] = r["result"]
    by_domain = defaultdict(list)
    for path in sorted((a.simulation / VARIANTS[0]).glob("*.json")):
        if not all((a.simulation / v / path.name).exists() for v in VARIANTS):
            continue
        dialogues = {v: json.loads((a.simulation / v / path.name).read_text()) for v in VARIANTS}
        if not a.turns[0] <= len(dialogues[VARIANTS[0]]["turns"]) <= a.turns[1]:
            continue
        domains = sorted(d for d, g in goals[path.stem]["meta"].items() if d != "profile" and g)
        if len(by_domain[domains[0]]) >= a.per_domain:
            continue
        entry = {"id": path.stem, "domains": domains, "goal": goals[path.stem]["texto"], "variants": {}}
        for v in VARIANTS:
            turns = [{"speaker": t["speaker"], "text": t["text"]} for t in dialogues[v]["turns"]]
            j = judged.get((v, path.stem), {})
            entry["variants"][v] = {"turns": turns, "steps": steps(turns, traces[v], allowed),
                                    "judge": {"goal": j.get("meta"), "quality": j.get("calidad")}}
        by_domain[domains[0]].append(entry)
    data = {"structure": structure, "goals": [g for d in sorted(by_domain) for g in by_domain[d]],
            "data": json.loads((a.domain / "domain.json").read_text()).get("data")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False))
    print(f"{len(data['goals'])} goals, {len(structure['states'])} states -> {a.out}")


if __name__ == "__main__":
    main()
