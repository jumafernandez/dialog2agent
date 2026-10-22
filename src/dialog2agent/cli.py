"""Command line interface: ``dialog2agent chat``, ``evaluate``, ``report`` and ``build``.

``evaluate`` can run as several processes, one per accelerator, with ``torch.distributed.run``: each process takes
the dialogues of its rank and writes its own file of predictions, so no communication between processes is needed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def default_device() -> str:
    """The accelerator of the local rank of the process, or the CPU."""
    import torch
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        return f"cuda:{local}"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local)
        return f"xpu:{local}"
    return "cpu"


def load(args):
    from .agent import Agent
    device = args.device or default_device()
    if getattr(args, "require_accelerator", False) and device == "cpu":
        raise SystemExit(f"no accelerator visible on {os.uname().nodename}")
    print(f"loading {args.domain} on {device} ({os.uname().nodename})", flush=True)
    return Agent.from_pretrained(args.domain, device=device, base_model=args.base_model,
                                 turn_encoder=args.turn_encoder)


def chat(args) -> None:
    agent = load(args)
    session = agent.new_session()
    print("Type your message (empty line to quit).")
    while (text := input("user> ").strip()):
        r = agent.turn(session, text)
        print(f"[state {r.state} · {r.action.name} · p={r.options[0].probability:.2f}]")
        print(f"agent> {r.text}")


def evaluate(args) -> None:
    from .evaluation import load_dialogues, predict
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    agent = load(args)
    dialogues = load_dialogues(args.dialogues)
    if args.max_dialogues:
        dialogues = dialogues[:args.max_dialogues]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    n = predict(agent, dialogues, out / f"predictions_{rank}.jsonl", write=not args.no_write, shard=rank,
                num_shards=world, log_every=50 if rank == 0 else 0)
    print(f"EVALUATE-OK rank {rank}: {n} new predictions", flush=True)


def report(args) -> None:
    from .evaluation import report as compute
    print(json.dumps(compute(sorted(Path(args.output).glob("predictions_*.jsonl"))), indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(prog="dialog2agent")
    sub = parser.add_subparsers(dest="command", required=True)

    def agent_options(p):
        p.add_argument("domain", help="directory of the domain artifact")
        p.add_argument("--device")
        p.add_argument("--base-model", help="replaces the base model of domain.json (for example, a local copy)")
        p.add_argument("--turn-encoder", help="replaces the turn encoder of domain.json (for example, a local copy)")

    p = sub.add_parser("chat", help="talk to an agent built from a domain artifact")
    agent_options(p)
    p.set_defaults(func=chat)
    p = sub.add_parser("evaluate", help="predict the system turns of a corpus with the real history")
    agent_options(p)
    p.add_argument("dialogues", help="JSON lines with dialogue_id and turns")
    p.add_argument("--output", required=True, help="directory of the predictions")
    p.add_argument("--no-write", action="store_true", help="decide the actions without writing the responses")
    p.add_argument("--max-dialogues", type=int, default=0)
    p.add_argument("--require-accelerator", action="store_true", help="stop if no accelerator is visible")
    p.set_defaults(func=evaluate)
    p = sub.add_parser("report", help="metrics of the predictions of a directory")
    p.add_argument("output")
    p.set_defaults(func=report)
    p = sub.add_parser("build", help="build a domain artifact from unannotated dialogues")
    p.set_defaults(func=lambda a: (_ for _ in ()).throw(NotImplementedError("'build' is in progress")))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
