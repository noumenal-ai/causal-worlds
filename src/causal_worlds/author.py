"""The Claude author adapter: natural-language prompt -> :class:`WorldSpec`, behind the seam.

This is the imperative shell's LLM edge. ``instructor`` forces the model to emit a validated
:class:`WorldSpecModel` (the pydantic boundary) with bounded re-ask, which we convert to the frozen
core IR. The provider SDKs are imported lazily in :func:`build_claude_author` so the package imports
(and CI runs) without the ``llm`` extra; the adapter logic itself takes an injected client and is
unit-tested with a fake — no API key required.

The author is deliberately a *different model family* than the Gemini judge (see
:mod:`causal_worlds.judge`): a world is never graded easy by the same brain that wrote it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from causal_worlds.serde import WorldSpecModel

if TYPE_CHECKING:
    import instructor

    from causal_worlds.schema import WorldSpec

DEFAULT_AUTHOR_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 4096
_MAX_RETRIES = 2  # instructor's bounded re-ask on a schema-invalid response

_SYSTEM_BASE = """\
You design small, fictional-but-internally-consistent CAUSAL OPERATIONS for a causal-discovery
benchmark. Output a world as a structural causal model: variables with roles, and a linear-plus-
Gaussian-noise mechanism per non-root variable.

Hard requirements:
- 5 to 9 variables. The graph MUST be acyclic.
- At least one OBSERVABLE CONTROLLABLE lever and one OBSERVABLE OUTCOME (the KPI).
- Effects must be recoverable: coefficients roughly 0.5-2.0 in magnitude, noise_scale around
  0.3, so signal dominates noise.

Definitions you will be asked to use:
- A HIDDEN confounder = a latent variable (hidden=true) that directly causes TWO OR MORE observed
  variables having NO direct edge between them (they correlate without causation — the trap).
- A REGIME flip = a binary disturbance named as 'regime' on a mechanism, with a key coefficient's
  SIGN FLIPPED in 'regime_terms' vs 'terms' (a lever's effect reverses between regimes)."""

# How much structural difficulty to inject — the complexity knob that spreads the benchmark.
_COMPLEXITY = {
    "easy": "Structure target: keep it transparent — a mostly-direct causal chain or two, with NO "
    "hidden confounder and NO regime flip. The world should be relatively easy to recover.",
    "standard": "Structure target: include exactly ONE hidden confounder and ONE regime flip.",
    "hard": "Structure target: make it genuinely deceptive — TWO OR MORE hidden confounders (each "
    "a common cause of 2+ observed variables) AND TWO regime flips. The structure must not be "
    "guessable from the variable names.",
}
_TEMPORAL_CLAUSE = (
    "This is a TEMPORAL operation that evolves over time. Use lagged terms (set a term's 'lag' to "
    "1 or 2) for delayed effects, and give at least one variable an autoregressive term (its own "
    "past, lag 1). Keep every autoregressive coefficient below 1 in magnitude (stationarity)."
)
_CLOSING = (
    "The world should be plausible for the described operation. These worlds are fictional; do not "
    "model any real system. Return ONLY the structured world."
)


def _system(complexity: str, *, temporal: bool = False) -> str:
    """Assemble the system brief for a complexity level (optionally temporal)."""
    if complexity not in _COMPLEXITY:
        msg = f"unknown complexity {complexity!r}; choose from {sorted(_COMPLEXITY)}"
        raise ValueError(msg)
    parts = [_SYSTEM_BASE, _COMPLEXITY[complexity]]
    if temporal:
        parts.append(_TEMPORAL_CLAUSE)
    parts.append(_CLOSING)
    return "\n\n".join(parts)


class ClaudeAuthor:
    """Authors a :class:`WorldSpec` from prose via an injected ``instructor`` Claude client."""

    def __init__(  # noqa: PLR0913 — keyword-only construction knobs for the live adapter
        self,
        client: instructor.Instructor,
        model: str = DEFAULT_AUTHOR_MODEL,
        *,
        complexity: str = "standard",
        temporal: bool = False,
        max_tokens: int = _MAX_TOKENS,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        """Store the client, model, and the system brief for the chosen complexity level."""
        self._client = client
        self._model = model
        self._system = _system(complexity, temporal=temporal)
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def author(self, prompt: str, *, feedback: str | None = None) -> WorldSpec:
        """Author a world from ``prompt``; ``feedback`` re-asks after a failed gate."""
        # `messages` is provider-shaped (OpenAI-style dicts); the precise param type lives in the
        # provider SDK, so it crosses the seam as Any rather than leaking those types into our code.
        messages: Any = self._messages(prompt, feedback)
        spec_model: WorldSpecModel = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            max_retries=self._max_retries,
            response_model=WorldSpecModel,
            messages=messages,
        )
        return spec_model.to_spec()

    def _messages(self, prompt: str, feedback: str | None) -> list[dict[str, str]]:
        """Build the chat messages: the standing system brief, the prompt, and any gate feedback."""
        user = f"Operation to model:\n{prompt}"
        if feedback is not None:
            user = f"{user}\n\nRevise your previous world. {feedback}"
        return [{"role": "system", "content": self._system}, {"role": "user", "content": user}]


def build_claude_author(
    model: str = DEFAULT_AUTHOR_MODEL,
    *,
    complexity: str = "standard",
    temporal: bool = False,
    api_key: str | None = None,
) -> ClaudeAuthor:  # pragma: no cover - real provider wiring, exercised only in live runs
    """Construct a live Claude author; needs the ``llm`` extra and an Anthropic API key in env."""
    import instructor  # noqa: PLC0415 - lazy: the provider SDK is an optional `llm` extra
    from anthropic import Anthropic  # noqa: PLC0415

    client = instructor.from_anthropic(Anthropic(api_key=api_key))
    return ClaudeAuthor(client, model, complexity=complexity, temporal=temporal)
