"""The agent: perceives the dialogue, decides the next action over the induced structure, acts and writes.

``Agent.respond`` computes the next system turn of a dialogue without changing any state, and ``Agent.turn`` applies
it to an ongoing session. The evaluation calls ``respond`` on the real dialogues and the user simulation and the web
demo call ``turn``, so that what is shown is what was evaluated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .backends import LLMBackend
from .decision import Decider, LLMScoringDecider
from .extraction import ConstraintExtractor
from .memory import SituationMemory
from .prompting import PromptConfig, action_prefix, messages, parse_output
from .representations import ContextEncoder, DialogueEncoder, TurnEncoder
from .scope import Scope, ScopeCheck
from .structure import InducedStructure
from .tools import ToolRegistry
from .types import Action, Option, Session, Turn, TurnResult

MAX_CONTEXT_TURNS = 12


class Agent:
    """A task-oriented dialogue agent that decides over an induced structure with a conversational memory.

    Args:
        structure: induced states, actions and allowed actions.
        memory: situations of other dialogues.
        backend: language model that scores the actions and writes the response.
        turn_encoder: Dialog2Flow turn encoder (``e_t``).
        context_encoder: TRACE contextual encoder (``h_t``), used to locate the state.
        dialogue_encoder: TRACE-Dialog encoder (``h_CLS``), used to query the memory.
        decider: how the action is chosen; by default, the language model scores the allowed actions.
        tools: tools triggered by the decided action.
        k: number of retrieved situations.
        conditioned: whether retrieval is conditioned by the structure.
        prompt: components of the prompt.
        decision: ``"score"`` (the decider chooses among the allowed actions and the model writes after the chosen
            one) or ``"generate"`` (the model writes the action and the response, as the reference of the
            evaluation; the declared action is projected to the closest induced action).
        retries: with ``"generate"``, how many sampled outputs are tried when the declared action is not allowed.
        scope: thresholds that tell whether a user turn is within the scope of the domain (see ``Scope``).
        extractor: writes the constraints of the user when the decided action calls a tool and the caller does not
            give them (see ``dialog2agent.extraction``).
    """

    def __init__(self, structure: InducedStructure, memory: SituationMemory, backend: LLMBackend,
                 turn_encoder: TurnEncoder, context_encoder: ContextEncoder, dialogue_encoder: DialogueEncoder,
                 decider: Decider | None = None, tools: ToolRegistry | None = None, k: int = 10,
                 conditioned: bool = True, prompt: PromptConfig | None = None, decision: str = "score",
                 retries: int = 3, scope: Scope | None = None, extractor: ConstraintExtractor | None = None):
        self.structure, self.memory, self.backend = structure, memory, backend
        self.turn_encoder, self.context_encoder, self.dialogue_encoder = turn_encoder, context_encoder, dialogue_encoder
        self.decider = decider or LLMScoringDecider(backend, structure)
        self.tools = tools or ToolRegistry()
        self.k, self.conditioned, self.prompt = k, conditioned, prompt or PromptConfig()
        if decision not in ("score", "generate"):
            raise ValueError(f"unknown decision mode: {decision}")
        self.decision, self.retries = decision, retries
        self.scope, self.extractor = scope, extractor
        self.metadata: dict = {}  # configuration of the domain artifact, including where its data comes from

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> Agent:
        """Load an agent from a domain artifact, a local directory or a repository of the Hugging Face Hub (see
        ``dialog2agent.domain``)."""
        from .domain import load_agent
        return load_agent(path, **kwargs)

    def check_scope(self, text: str) -> ScopeCheck:
        """Whether a user turn is within the scope of the domain, with the value of each signal."""
        if self.scope is None:
            raise ValueError("the agent has no scope; calibrate one with Scope.calibrate")
        return self.scope.check(self, text)

    def new_session(self) -> Session:
        return Session()

    def respond(self, turns: list[Turn], constraints: dict[str, dict[str, str]] | None = None,
                write: bool = True) -> TurnResult:
        """Decide and write the next system turn of a dialogue that ends with a user turn.

        With ``write=False`` only the action is decided and the text of the result is empty.
        """
        if not turns or turns[-1].speaker != "user":
            raise ValueError("the dialogue must end with a user turn")
        e = self.turn_encoder.encode([t.text for t in turns])
        h = self.context_encoder.encode(turns, e)
        state = int(self.structure.state(e[-1], h[-1]))
        allowed = self.structure.allowed(state)

        h_cls = self.dialogue_encoder.encode(turns, e)
        query = self.memory.represent(e[-1], np.asarray(h_cls).reshape(-1))
        situations = self.memory.search(query, self.k, state=state if self.conditioned else None,
                                        allowed=allowed if self.conditioned else None)

        context = list(turns[-MAX_CONTEXT_TURNS:])
        while context and context[0].speaker != "user":
            context = context[1:]
        names = sorted({self.structure.names[a] for a in allowed})
        chat = messages(context, situations, names, self.prompt)
        if self.decision == "generate":
            action, text = self._generate(chat, allowed)
            return TurnResult(text=text, action=action, state=state,
                              options=[Option(action=action, probability=1.0, score=0.0)], situations=situations)
        options = self.decider.decide(chat, state, allowed, situations)
        action = options[0].action

        tool_calls = []
        tool = self.tools.for_action(action)
        if tool is not None and not constraints and self.extractor is not None:
            # the extractor writes first the service of the current request, which is the one the tool acts on
            constraints = dict(list(self.extractor.extract(list(turns)).items())[:1])
        if tool is not None and constraints:
            for domain, values in constraints.items():
                if values or tool.name != "search":  # a search needs constraints; a taxi booking does not
                    tool_calls.append(tool(domain=domain, constraints=values))
        if tool_calls:
            chat = messages(context, situations, names, self.prompt,
                            tool_results=[f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)}) -> "
                                          f"{json.dumps(c.result, ensure_ascii=False)}" for c in tool_calls])
        text = self.backend.generate(chat, prefix=action_prefix(action.name)).strip() if write else ""
        return TurnResult(text=text, action=action, state=state, options=options, situations=situations,
                          tool_calls=tool_calls)

    def _generate(self, chat: list[dict], allowed: list[int]) -> tuple[Action, str]:
        """Decision by generation, as in the evaluation of the agent: the model writes the action and the response,
        the declared action is projected to the closest induced action and, if that action is not allowed, the
        output is sampled again up to ``retries`` times, keeping the first output when no sample is allowed."""
        first = None
        for attempt in range(self.retries + 1):
            name, text = parse_output(self.backend.generate(chat, sample=attempt > 0))
            region = int(self.structure.system_action(self.turn_encoder.encode([name]))[0]) if name else -1
            first = first or (name, text, region)
            if region in allowed:
                break
        else:
            name, text, region = first
        return (self.structure.action(region) if region >= 0 else Action(id=-1, name=name or "")), text

    def turn(self, session: Session, user_text: str) -> TurnResult:
        """Process a user turn of a session and return the response with the full trace of the decision."""
        session.turns.append(Turn("user", user_text))
        result = self.respond(session.turns, constraints=session.constraints)
        session.turns.append(Turn("system", result.text))
        session.states.append(result.state)
        session.actions.append(result.action)
        return result
