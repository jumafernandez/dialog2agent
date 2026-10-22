"""The agent: perceives the dialogue, decides the next action over the induced structure, acts and writes.

The same ``Agent.turn`` is used by the evaluation, by the user simulation and by the web demo, so that what is shown
is what was evaluated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends import LLMBackend
from .decision import Decider, LLMScoringDecider
from .memory import SituationMemory
from .prompting import PromptConfig, action_prefix, messages
from .representations import ContextEncoder, DialogueEncoder, TurnEncoder
from .structure import InducedStructure
from .tools import ToolRegistry
from .types import Session, Turn, TurnResult

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
    """

    def __init__(self, structure: InducedStructure, memory: SituationMemory, backend: LLMBackend,
                 turn_encoder: TurnEncoder, context_encoder: ContextEncoder, dialogue_encoder: DialogueEncoder,
                 decider: Decider | None = None, tools: ToolRegistry | None = None, k: int = 10,
                 conditioned: bool = True, prompt: PromptConfig | None = None):
        self.structure, self.memory, self.backend = structure, memory, backend
        self.turn_encoder, self.context_encoder, self.dialogue_encoder = turn_encoder, context_encoder, dialogue_encoder
        self.decider = decider or LLMScoringDecider(backend, structure)
        self.tools = tools or ToolRegistry()
        self.k, self.conditioned, self.prompt = k, conditioned, prompt or PromptConfig()

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> Agent:
        """Load an agent from a domain artifact (see ``dialog2agent.domain``)."""
        from .domain import load_agent
        return load_agent(path, **kwargs)

    def new_session(self) -> Session:
        return Session()

    def turn(self, session: Session, user_text: str) -> TurnResult:
        """Process a user turn and return the response with the full trace of the decision."""
        session.turns.append(Turn("user", user_text))
        e = self.turn_encoder.encode([t.text for t in session.turns])
        h = self.context_encoder.encode(session.turns, e)
        state = int(self.structure.state(e[-1], h[-1]))
        allowed = self.structure.allowed(state)

        h_cls = self.dialogue_encoder.encode(session.turns, e)
        query = self.memory.represent(e[-1], np.asarray(h_cls).reshape(-1))
        situations = self.memory.search(query, self.k, state=state if self.conditioned else None,
                                        allowed=allowed if self.conditioned else None)

        context = session.turns[-MAX_CONTEXT_TURNS:]
        while context and context[0].speaker != "user":
            context = context[1:]
        names = list(dict.fromkeys(self.structure.names[a] for a in allowed))
        chat = messages(context, situations, names, self.prompt)
        options = self.decider.decide(chat, state, allowed, situations)
        action = options[0].action

        tool_calls = []
        tool = self.tools.for_action(action.name)
        if tool is not None and session.constraints:
            for domain, constraints in session.constraints.items():
                tool_calls.append(tool(domain=domain, constraints=constraints))
        if tool_calls:
            chat = messages(context, situations, names, self.prompt,
                            tool_results=[f"{c.name}({c.arguments}) -> {c.result}" for c in tool_calls])
        text = self.backend.generate(chat, prefix=action_prefix(action.name)).strip()

        session.turns.append(Turn("system", text))
        session.states.append(state)
        session.actions.append(action)
        return TurnResult(text=text, action=action, state=state, options=options, situations=situations,
                          tool_calls=tool_calls)
