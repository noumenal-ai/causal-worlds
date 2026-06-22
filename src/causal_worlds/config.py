"""Runtime configuration, loaded and validated at the boundary (pydantic-settings).

Read from env vars prefixed ``CAUSAL_WORLDS_`` (or a local ``.env``); frozen once loaded. Secrets
(LLM/observability keys) arrive here once the v0.2 author/judge land — env only, never code or logs.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    langfuse_enabled: bool = False  # the observability seam is off until configured (v0.2)
