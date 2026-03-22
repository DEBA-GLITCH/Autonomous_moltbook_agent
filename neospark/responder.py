from __future__ import annotations

from .config import Settings
from .llm import LLMClient
from .prompts import SYSTEM_PERSONA_PROMPT


def handle_interactive_request(data: dict) -> dict:
    """
    Compatibility handler used by skill runtime.

    This path keeps your previous request/response contract:
    input/context/agents -> one LLM response.
    """
    settings = Settings.from_env()
    llm = LLMClient(settings)

    user_input = data.get("input", "")
    context = data.get("context", "")
    other_agents = data.get("agents", [])

    agent_context = ""
    if other_agents:
        lines = "\n".join(f"- {agent}" for agent in other_agents)
        agent_context = f"\nOther agent responses:\n{lines}\n"

    prompt = (
        f"Context:\n{context}\n\n"
        f"User Input:\n{user_input}\n"
        f"{agent_context}"
    )

    response = llm.chat(prompt, system_prompt=SYSTEM_PERSONA_PROMPT)
    return {"agent": "NeoSparkCore", "response": response}
