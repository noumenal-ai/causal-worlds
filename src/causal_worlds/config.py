"""Runtime configuration, loaded and validated at the boundary (pydantic-settings).

Read from env vars prefixed ``CAUSAL_WORLDS_`` (or a local ``.env``); frozen once loaded. Provider
API keys are NOT held here — the SDKs read their own standard env vars (``ANTHROPIC_API_KEY``,
``GEMINI_API_KEY``); we never copy a secret into code, settings, or logs.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from causal_worlds.author import DEFAULT_AUTHOR_MODEL
from causal_worlds.judge import DEFAULT_JUDGE_MODEL


class Settings(BaseSettings):
    """Runtime settings for a causal-worlds run."""

    model_config = SettingsConfigDict(
        env_prefix="CAUSAL_WORLDS_",
        env_file=".env",
        frozen=True,
        extra="ignore",
    )

    seed: int = 0
    discoverer_n: int = 8000  # samples per environment for the reference grader
    author_model: str = DEFAULT_AUTHOR_MODEL  # Claude family (independent of the judge)
    judge_model: str = DEFAULT_JUDGE_MODEL  # Gemini family (independent of the author)
    max_attempts: int = 3  # author re-asks before a prompt is abandoned
    bundle_rows: int = 2000  # rows sampled into each persisted world's data.npz
    langfuse_enabled: bool = False  # the observability seam is off until configured
