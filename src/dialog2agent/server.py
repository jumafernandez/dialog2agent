"""Web service of an agent: the same ``Agent.respond`` that is evaluated, behind an HTTP API.

The service keeps no state between requests. The client sends the dialogue so far, which ends with a user turn, and
receives the next system turn with the full trace of the decision, so that any replica of the service can answer
any request and replicas can be started and stopped at any time.

Endpoints:
    GET  /health      whether the agent is loaded
    GET  /about       library version and attribution of the data of the domain artifact
    GET  /structure   induced actions and states, with the allowed actions of each state, for visualization
    POST /respond     next system turn of a dialogue, with its trace
    POST /scope       whether a user turn is within the scope of the domain (when the artifact declares a scope)

The agent is also served through the chat completions API of OpenAI (``GET /v1/models`` and
``POST /v1/chat/completions``), so that any OpenAI-compatible client, such as the user simulators of SDialog, can
converse with it. System messages are ignored, user and assistant messages are the turns of the dialogue, and the
response carries the trace of the decision in an additional ``trace`` field.
"""
# Without "from __future__ import annotations": FastAPI reads the request models from the annotations of the
# endpoints, which are defined inside create_app.
import json
import threading
import time
import uuid
from dataclasses import asdict
from typing import Literal

MAX_TURNS = 60
MAX_CHARS = 1000


def result_to_dict(result, structure=None) -> dict:
    """A ``TurnResult`` as JSON-serializable data; with ``structure``, also the allowed actions of the state."""
    allowed = ([{"id": a, "name": structure.names[a]} for a in structure.allowed(result.state)]
               if structure is not None else None)
    return {"text": result.text, "state": result.state, "allowed": allowed,
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


def message_text(content) -> str:
    """Text of an OpenAI message, whose content is a string or a list of parts."""
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def create_app(agent, allowed_origins: list[str] | None = None, log_path: str | None = None,
               model_id: str = "dialog2agent"):
    """FastAPI application that serves ``agent``. Requests are answered one at a time, since the agent holds a
    single model on the accelerator. With ``log_path``, every response is appended to that file as a JSON line with
    the dialogue and the trace of the decision."""
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
        decision: Literal["score", "generate"] | None = None

    class ScopeIn(BaseModel):
        text: str = Field(max_length=MAX_CHARS)

    class MessageIn(BaseModel):
        role: str
        content: str | list | None = None

    class ChatIn(BaseModel):
        model_config = {"extra": "allow"}
        model: str | None = None
        messages: list[MessageIn] = Field(min_length=1, max_length=2 * MAX_TURNS)

    app = FastAPI(title="dialog2agent", description=__doc__)
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins or ["*"], allow_methods=["GET", "POST"],
                       allow_headers=["*"])
    lock = threading.Lock()
    from . import __version__
    about = {"library": "dialog2agent", "version": __version__, "data": getattr(agent, "metadata", {}).get("data"),
             "decisions": ["score", "generate"], "default_decision": agent.decision,
             "scope": agent.scope.to_dict() if getattr(agent, "scope", None) is not None else None}
    structure = {**structure_to_dict(agent.structure), "data": about["data"]}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/about")
    def get_about() -> dict:
        return about

    @app.get("/structure")
    def get_structure() -> dict:
        return structure

    def run(turns: list, write: bool = True, request: str = "respond", decision: str | None = None) -> dict:
        if not turns or turns[-1].speaker != "user":
            raise HTTPException(status_code=422, detail="the dialogue must end with a user turn")
        with lock:  # the decision mode of a request applies only to that request
            default, agent.decision = agent.decision, decision or agent.decision
            try:
                result = result_to_dict(agent.respond(turns, write=write), agent.structure)
            finally:
                agent.decision = default
            result["decision"] = decision or default
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"time": time.time(), "request": request,
                                        "turns": [asdict(t) for t in turns], "result": result},
                                       ensure_ascii=False) + "\n")
        return result

    @app.post("/respond")
    def respond(body: RespondIn) -> dict:
        return run([Turn("user" if t.speaker == "user" else "system", t.text) for t in body.turns], body.write,
                   decision=body.decision)

    @app.post("/scope")
    def scope(body: ScopeIn) -> dict:
        if about["scope"] is None:
            raise HTTPException(status_code=404, detail="the domain artifact declares no scope")
        with lock:
            return asdict(agent.check_scope(body.text))

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "dialog2agent"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatIn) -> dict:
        turns = [Turn("user" if m.role == "user" else "system", message_text(m.content)[:MAX_CHARS])
                 for m in body.messages if m.role in ("user", "assistant")][-MAX_TURNS:]
        result = run(turns, request="chat")
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "trace": result}

    return app
