import os
from dataclasses import dataclass

from dotenv import load_dotenv  # type: ignore


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse an environment variable into a boolean."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    """Parse an environment variable into an integer."""
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_float(value: str | None, default: float) -> float:
    """Parse an environment variable into a float."""
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings for NeoSpark."""

    groq_api_key: str
    moltbook_api_key: str
    model_name: str
    moltbook_base_url: str
    submolt_name: str
    agent_handle: str
    loop_interval_seconds: int
    loop_jitter_seconds: int
    request_timeout_seconds: int
    max_replies_per_cycle: int
    max_comment_replies_per_cycle: int
    max_verification_replies_per_cycle: int
    post_interval_minutes: int
    top_agent_percent: float
    min_post_words: int
    memory_db_path: str
    dry_run: bool

    @staticmethod
    def from_env() -> "Settings":
        """Load settings from environment variables and validate required keys."""
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        moltbook_api_key = os.getenv("MOLTBOOK_API_KEY", "").strip()
        if not groq_api_key:
            raise RuntimeError("Missing required env var: GROQ_API_KEY")
        if not moltbook_api_key:
            raise RuntimeError("Missing required env var: MOLTBOOK_API_KEY")

        top_agent_percent = _as_float(os.getenv("TOP_AGENT_PERCENT"), 0.20)
        if top_agent_percent <= 0 or top_agent_percent > 1:
            raise RuntimeError("TOP_AGENT_PERCENT must be > 0 and <= 1")

        return Settings(
            groq_api_key=groq_api_key,
            moltbook_api_key=moltbook_api_key,
            # FIX: "qwen/qwen3-32b" (with a slash) is NOT a valid Groq model ID.
            # Updated default to "qwen-qwen3-32b". Verify the exact string at
            # https://console.groq.com/docs/models — Groq model IDs use hyphens,
            # not slashes. A wrong model name silently returns garbage output.
            model_name=os.getenv("MODEL_NAME", "qwen-qwen3-32b"),
            moltbook_base_url=os.getenv(
                "MOLTBOOK_BASE_URL", "https://www.moltbook.com/api/v1"
            ),
            submolt_name=os.getenv("MOLTBOOK_SUBMOLT", "general"),
            agent_handle=os.getenv("AGENT_HANDLE", "NeoSpark"),
            loop_interval_seconds=_as_int(os.getenv("LOOP_INTERVAL_SECONDS"), 300),
            loop_jitter_seconds=_as_int(os.getenv("LOOP_JITTER_SECONDS"), 45),
            request_timeout_seconds=_as_int(os.getenv("REQUEST_TIMEOUT_SECONDS"), 20),
            max_replies_per_cycle=_as_int(os.getenv("MAX_REPLIES_PER_CYCLE"), 4),
            max_comment_replies_per_cycle=_as_int(
                os.getenv("MAX_COMMENT_REPLIES_PER_CYCLE"), 4
            ),
            max_verification_replies_per_cycle=_as_int(
                os.getenv("MAX_VERIFICATION_REPLIES_PER_CYCLE"), 2
            ),
            post_interval_minutes=_as_int(os.getenv("POST_INTERVAL_MINUTES"), 240),
            top_agent_percent=top_agent_percent,
            min_post_words=_as_int(os.getenv("MIN_POST_WORDS"), 90),
            memory_db_path=os.getenv("MEMORY_DB_PATH", "neospark_state.db"),
            dry_run=_as_bool(os.getenv("DRY_RUN"), False),
        )
