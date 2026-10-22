"""Command line interface: ``dialog2agent chat``, ``dialog2agent build`` and ``dialog2agent evaluate``."""
from __future__ import annotations

import argparse


def chat(args) -> None:
    from .agent import Agent
    agent = Agent.from_pretrained(args.domain, device=args.device)
    session = agent.new_session()
    print("Type your message (empty line to quit).")
    while (text := input("user> ").strip()):
        r = agent.turn(session, text)
        print(f"[state {r.state} · {r.action.name} · p={r.options[0].probability:.2f}]")
        print(f"agent> {r.text}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="dialog2agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("chat", help="talk to an agent built from a domain artifact")
    p.add_argument("domain")
    p.add_argument("--device")
    p.set_defaults(func=chat)
    for name, text in (("build", "build a domain artifact from unannotated dialogues"),
                       ("evaluate", "evaluate an agent on a corpus")):
        q = sub.add_parser(name, help=text)
        q.set_defaults(func=lambda a, n=name: (_ for _ in ()).throw(NotImplementedError(f"'{n}' is in progress")))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
