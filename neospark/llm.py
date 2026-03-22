import json
import re
from typing import Any

from groq import Groq  # type: ignore

from .config import Settings
from .prompts import SYSTEM_PERSONA_PROMPT


class LLMClient:
    """Thin wrapper around Groq chat completions with basic cleanup helpers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        # FIX: Ensure the model name in your .env matches an actual Groq model ID.
        # Valid examples: "llama-3.3-70b-versatile", "llama3-70b-8192",
        # "qwen-qwen3-32b" (check console.groq.com/docs/models for current names).
        # "qwen/qwen3-32b" (with a slash) is NOT a valid Groq model ID and will
        # cause the API to fail or silently fall back, producing garbage output.
        self.client = Groq(api_key=settings.groq_api_key)

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip model-internal think tags and surrounding whitespace."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def chat(
        self,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
        system_prompt: str = SYSTEM_PERSONA_PROMPT,
    ) -> str:
        """Run a plain text completion."""
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        return self.clean_text(content)

    def chat_json(
        self,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 700,
        system_prompt: str = SYSTEM_PERSONA_PROMPT,
    ) -> dict[str, Any]:
        """Run a completion and parse JSON output with a defensive fallback."""
        raw = self.chat(
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
