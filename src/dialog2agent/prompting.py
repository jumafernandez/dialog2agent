"""Input and output format of the language model.

The input is a chat with a system message that contains the instructions, the situations retrieved from the memory
and the names of the allowed actions, followed by the turns of the ongoing dialogue. The output is the action followed
by the response (``Action: ...\\nResponse: ...``). The texts must match the ones used to adapt the model, so they are
kept verbatim from the adaptation of the published adapters.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Situation, Turn

INSTRUCTION = "You are a task-oriented dialogue agent. Write the next turn of the system in the conversation below."
INSTRUCTION_ACTION = (" First declare the action of your turn on a line starting with 'Action:', and then write the "
                      "turn on a line starting with 'Response:'.")
INSTRUCTION_NAMES = " Use the action names shown below."
INSTRUCTION_MEMORY = " Similar situations from other conversations show what the system did next."
INSTRUCTION_GUIDE = " The allowed actions are the ones the system usually performs in the current state."
INSTRUCTION_TOOLS = " Use the information returned by the tools when you mention entities or their attributes."
MAX_CHARS = 300


@dataclass
class PromptConfig:
    """Which components enter the prompt (they correspond to the ablations of the agent)."""

    memory: bool = True
    guide: bool = True
    action: bool = True


def truncate(text: str, n: int = MAX_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def system_message(situations: list[Situation], allowed: list[str], config: PromptConfig,
                   tool_results: list[str] | None = None) -> str:
    text = INSTRUCTION + (INSTRUCTION_ACTION if config.action else "")
    if config.action and ((config.memory and situations) or (config.guide and allowed)):
        text += INSTRUCTION_NAMES
    if config.memory and situations:
        text += INSTRUCTION_MEMORY
    if config.guide and allowed:
        text += INSTRUCTION_GUIDE
    if tool_results:
        text += INSTRUCTION_TOOLS
    if config.memory and situations:
        lines = []
        for i, s in enumerate(situations, 1):
            action = f" | Action: {s.action}" if config.action else ""
            lines.append(f"{i}. User: {truncate(s.user)}{action} | System: {truncate(s.response)}")
        text += "\n\nSimilar situations:\n" + "\n".join(lines)
    if config.guide and allowed:
        text += "\n\nAllowed actions in the current state:\n" + "\n".join(f"- {a}" for a in allowed)
    if tool_results:
        text += "\n\nTool results:\n" + "\n".join(tool_results)
    return text


def messages(turns: list[Turn], situations: list[Situation], allowed: list[str], config: PromptConfig,
             tool_results: list[str] | None = None) -> list[dict]:
    chat = [{"role": "user" if t.speaker == "user" else "assistant", "content": t.text} for t in turns]
    return [{"role": "system", "content": system_message(situations, allowed, config, tool_results)}] + chat


def action_prefix(name: str) -> str:
    """Beginning of the output once the action is fixed."""
    return f"Action: {name}\nResponse:"


def parse_output(text: str) -> tuple[str | None, str]:
    """Split a generation into the declared action (None if missing) and the response."""
    action, response = None, text.strip()
    if text.lstrip().startswith("Action:"):
        head, _, rest = text.lstrip()[len("Action:"):].partition("\n")
        action, response = head.strip(), rest.strip()
        if response.startswith("Response:"):
            response = response[len("Response:"):].strip()
    return action, response
