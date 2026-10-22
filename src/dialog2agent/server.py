"""Web service of an agent: the same ``Agent.respond`` that is evaluated, behind an HTTP API.

The service keeps no state between requests. The client sends the dialogue so far, which ends with a user turn, and
receives the next system turn with the full trace of the decision, so that any replica of the service can answer
any request and replicas can be started and stopped at any time.

Endpoints:
    GET  /health      whether the agent is loaded
    GET  /structure   induced actions and states, with the allowed actions of each state, for visualization
    POST /respond     next system turn of a dialogue, with its trace
"""
# Without "from __future__ import annotations": FastAPI reads the request models from the annotations of the
# endpoints, which are defined inside create_app.
import threading
from dataclasses import asdict

MAX_TURNS = 60
MAX_CHARS = 1000


def result_to_dict(result) -> dict:
    """A ``TurnResult`` as JSON-serializable data."""
    return {"text": result.text, "state": result.state,
            "action": {"id": result.action.id, "name": result.action.name},
            "options": [{"id": o.action.id, "name": o.action.name, "probability": o.probability}
                        for o in result.options],
            "situations": [asdict(s) for s in result.situations],
            "tool_calls": [asdict(c) for c in result.tool_calls]}


def structure_to_dict(structure) -> dict:
    """Actions and states of the induced structure, with the allowed actions of each state and their probability."""
    return {"actions": [{"id": i, "name": n} for i, n in enumerate(structure.names)],
            "states": [{"id": s, "allowed": [{"id": a, "probability": float(structure.prior(s)[a])}
                                             for a in structure.allowed(s)]}
                       for s in range(len(structure.state_centers))]}


def create_app(agent, allowed_origins: list[str] | None = None):
    """FastAPI application that serves ``agent``. Requests are answered one at a time, since the agent holds a
    single model on the accelerator."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    from .types import Turn

    class TurnIn(BaseModel):
        speaker: str
        text: str = Field(max_length=MAX_CHARS)

    class RespondIn(BaseModel):
        turns: list[TurnIn] = Field(min_length=1, max_length=MAX_TURNS)
        write: bool = True

    app = FastAPI(title="dialog2agent", description=__doc__)
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins or ["*"], allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    lock = threading.Lock()
    structure = structure_to_dict(agent.structure)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/structure")
    def get_structure() -> dict:
        return structure

    @app.post("/respond")
    def respond(body: RespondIn) -> dict:
        turns = [Turn("user" if t.speaker == "user" else "system", t.text) for t in body.turns]
        if turns[-1].speaker != "user":
            raise HTTPException(status_code=422, detail="the dialogue must end with a user turn")
        with lock:
            return result_to_dict(agent.respond(turns, write=body.write))

    return app
