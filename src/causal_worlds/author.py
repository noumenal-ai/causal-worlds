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

_SYSTEM = """\
You design small, fictional-but-internally-consistent CAUSAL OPERATIONS for a causal-discovery
benchmark. Output a world as a structural causal model: variables with roles, and a linear-plus-
Gaussian-noise mechanism per non-root variable.

Hard requirements:
- 5 to 9 variables. The graph MUST be acyclic.
- At least one OBSERVABLE CONTROLLABLE lever and one OBSERVABLE OUTCOME (the KPI).
- Effects must be recoverable: coefficients roughly 0.5-2.0 in magnitude, noise_scale around
  0.3, so signal dominates noise.

What makes a GOOD (hard, non-cliché) world — include BOTH:
- A HIDDEN confounder: a latent variable (hidden=true) that directly causes TWO OR MORE observed
  variables that have NO direct edge between them. This makes those two observed variables
  correlate without one causing the other — the trap a naive method falls into.
- A REGIME flip: pick a binary disturbance variable as 'regime' on one mechanism; give
  'regime_terms' the SAME parents as 'terms' but with a key coefficient's SIGN FLIPPED (or
  rescaled). A lever's effect should reverse between regimes, so it can't be guessed from names.

The world should be plausible for the described operation, but its causal STRUCTURE must NOT be
obvious from the names — surprise is the point. These worlds are fictional; do not model any real
system. Return ONLY the structured world.\
"""


class ClaudeAuthor:
    """Authors a :class:`WorldSpec` from prose via an injected ``instructor`` Claude client."""

    def __init__(
        self,
        client: instructor.Instructor,
        model: str = DEFAULT_AUTHOR_MODEL,
        *,
        max_tokens: int = _MAX_TOKENS,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        """Store the client and generation settings (the client is constructed at the edge)."""
        self._client = client
        self._model = model
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

    @staticmethod
    def _messages(prompt: str, feedback: str | None) -> list[dict[str, str]]:
        """Build the chat messages: the standing system brief, the prompt, and any gate feedback."""
        user = f"Operation to model:\n{prompt}"
        if feedback is not None:
            user = f"{user}\n\nRevise your previous world. {feedback}"
        return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def build_claude_author(
    model: str = DEFAULT_AUTHOR_MODEL, *, api_key: str | None = None
) -> ClaudeAuthor:  # pragma: no cover - real provider wiring, exercised only in live runs
    """Construct a live Claude author; needs the ``llm`` extra and an Anthropic API key in env."""
    import instructor  # noqa: PLC0415 - lazy: the provider SDK is an optional `llm` extra
    from anthropic import Anthropic  # noqa: PLC0415

    client = instructor.from_anthropic(Anthropic(api_key=api_key))
    return ClaudeAuthor(client, model)
