"""The Gemini judge adapter: an independent second opinion, behind the seam.

The judge is deliberately a *different model family* than the Claude author. It does two jobs:

* :meth:`prior_edges` — guess the causal edges from variable **names and roles alone**, with no data
  and no sight of the mechanisms. The gap between this guess and the truth is the anti-cliché signal
  (T4): a world the judge nails from priors is a cliché.
* :meth:`faithfulness` — score how faithfully a spec represents the prose it was authored from.

Provider SDKs are imported lazily in :func:`build_gemini_judge`; the adapter logic takes an injected
client and is unit-tested with a fake, so the package imports and CI runs without the ``llm`` extra.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, Field

from causal_worlds.anonymize import anonymize_spec
from causal_worlds.schema import answer_key

if TYPE_CHECKING:
    import instructor

    from causal_worlds.protocols import Edges
    from causal_worlds.schema import WorldSpec

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

_M = TypeVar("_M", bound=BaseModel)  # the structured response model a judge call returns

# Provider overload is transient and common (Gemini 503 "high demand"). We retry it behind the seam
# so a spike doesn't bubble up and waste the (paid) upstream author call that produced the spec
# being judged. Matched on the error string to stay SDK-agnostic — any provider's overload is one.
_TRANSIENT_MARKERS = (
    "503",
    "unavailable",
    "overloaded",
    "resource_exhausted",
    "high demand",
    "429",
    "rate limit",
    "deadline",
    "timeout",
)


def _is_transient(exc: Exception) -> bool:
    """True if an error looks like a transient provider overload/throttle worth retrying."""
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class _Edge(BaseModel):
    """One guessed directed edge ``src -> dst``."""

    src: str = Field(description="Name of the cause variable.")
    dst: str = Field(description="Name of the effect variable.")


class _Prior(BaseModel):
    """The judge's prior guess at the causal structure, from names and roles alone."""

    edges: list[_Edge] = Field(
        description="Directed edges you'd expect with no data, from intuition."
    )


class _Faithfulness(BaseModel):
    """The judge's verdict on how well a spec matches the prose."""

    score: float = Field(
        ge=0.0, le=1.0, description="0 = unrelated, 1 = a faithful representation."
    )
    reason: str = Field(description="One sentence justifying the score.")


def _observed(spec: WorldSpec) -> list[tuple[str, str]]:
    """The observed ``(name, role)`` pairs the judge is allowed to see."""
    return [(v.name, v.role.value) for v in spec.variables if not v.hidden]


class GeminiJudge:
    """Scores worlds via an injected ``instructor`` Gemini client (an independent family)."""

    def __init__(
        self,
        client: instructor.Instructor,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        retries: int = 4,
        backoff: float = 2.0,
    ) -> None:
        """Store the client and model id (the client is constructed at the edge).

        ``retries`` and ``backoff`` bound transient-overload retry: attempt ``i`` waits
        ``backoff * 2**i`` seconds (default rides through a ~30s spike). ``backoff=0`` disables the
        wait — used in tests so they stay fast.
        """
        self._client = client
        self._model = model
        self._retries = retries
        self._backoff = backoff

    def _create(self, response_model: type[_M], prompt: str) -> _M:
        """Call the judge for ``response_model``, retrying transient overloads with backoff.

        A non-transient error is re-raised at once (no point retrying a bad request); a transient
        one is retried up to ``self._retries`` times, then re-raised so the caller still fails loud.
        """
        messages: Any = [{"role": "user", "content": prompt}]
        for attempt in range(self._retries + 1):
            try:
                result: _M = self._client.chat.completions.create(
                    model=self._model, response_model=response_model, messages=messages
                )
            except Exception as exc:
                if attempt == self._retries or not _is_transient(exc):
                    raise
                time.sleep(self._backoff * 2**attempt)
            else:
                return result
        msg = "unreachable: the retry loop always returns or raises"
        raise AssertionError(msg)  # pragma: no cover

    def prior_edges(self, spec: WorldSpec, *, blind: bool = False) -> Edges:
        """Guess the causal edges from priors alone (no data).

        Default: names + roles visible — the anti-cliché signal (how guessable a world is). With
        ``blind``, names are anonymized to ``X1..Xn`` and roles are hidden, so the guess can only
        use graph conventions — a control that should sit at chance (Caliper). Blind edges are
        mapped back to the real names, so the return is always over the spec's variables.
        """
        if blind:
            return self._blind_prior(spec)
        observed = _observed(spec)
        listing = "\n".join(f"- {name} ({role})" for name, role in observed)
        prompt = (
            "These are the observed variables of an operation, with their roles. Using ONLY "
            "general domain intuition (you have NO data and cannot see the true mechanism), list "
            f"the directed causal edges (cause -> effect) you would expect among them:\n{listing}"
        )
        return self._guess(prompt, {name for name, _ in observed})

    def _blind_prior(self, spec: WorldSpec) -> Edges:
        """Guess with anonymized names and NO roles, then map the edges back to the real names."""
        anon, mapping = anonymize_spec(spec)
        inverse = {anon_name: original for original, anon_name in mapping.items()}
        names = tuple(v.name for v in anon.variables if not v.hidden)
        listing = "\n".join(f"- {name}" for name in names)
        prompt = (
            "These are the (anonymized) observed variables of an operation. You have NO data, NO "
            "names, and NO roles — only the variable tokens. List any directed causal edges "
            f"(cause -> effect) you would guess among them:\n{listing}"
        )
        return frozenset(
            (inverse[src], inverse[dst]) for src, dst in self._guess(prompt, set(names))
        )

    def _guess(self, prompt: str, names: set[str]) -> Edges:
        """Ask the judge for prior edges and keep only valid, non-self edges among ``names``."""
        prior: _Prior = self._create(_Prior, prompt)
        return frozenset(
            (e.src, e.dst)
            for e in prior.edges
            if e.src in names and e.dst in names and e.src != e.dst
        )

    def faithfulness(self, prose: str, spec: WorldSpec) -> float:
        """Score in ``[0, 1]`` how faithfully ``spec`` represents ``prose``."""
        names = ", ".join(f"{name} ({role})" for name, role in _observed(spec))
        edges = answer_key(spec).edges
        drawn = ", ".join(f"{src}->{dst}" for src, dst in sorted(edges)) or "(no edges)"
        prompt = (
            f"Operation described:\n{prose}\n\n"
            f"A model proposes these observed variables: {names}\n"
            f"and these causal edges: {drawn}\n\n"
            "Score from 0 to 1 how faithfully this captures the described operation."
        )
        verdict: _Faithfulness = self._create(_Faithfulness, prompt)
        return max(0.0, min(1.0, verdict.score))


def build_gemini_judge(
    model: str = DEFAULT_JUDGE_MODEL, *, api_key: str | None = None
) -> GeminiJudge:  # pragma: no cover - real provider wiring, exercised only in live runs
    """Construct a live Gemini judge; needs the ``llm`` extra and a Gemini API key in the env."""
    import instructor  # noqa: PLC0415 - lazy: the provider SDK is an optional `llm` extra
    from google import genai  # noqa: PLC0415

    client = instructor.from_genai(genai.Client(api_key=api_key))
    return GeminiJudge(client, model)
