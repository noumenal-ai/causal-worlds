"""Conversational world elicitation — the stateful clarify loop that yields a ``WorldBrief``.

A one-shot prompt is underspecified, so interactive use runs a dialogue first: the elicitor asks the
*minimal* clarifying questions against the brief-completeness checklist, the accumulating brief is
shown, and generation fires only when the brief is complete (or the user says "go"). The elicitor
uses the **author** model — it helps the user *specify*, not grade, so the judge-independence rule
does not apply here.

``Session`` is the first-class state model (the CLI and a future web UI both drive it); the elicitor
adapter is stateless and re-reads the transcript each turn. Provider SDKs are imported lazily; the
adapter takes an injected client and is unit-tested with a fake, so the package imports keyless.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from causal_worlds.brief import WorldBrief, is_complete

if TYPE_CHECKING:
    from collections.abc import Sequence

    import instructor

    from causal_worlds.protocols import Elicitor

DEFAULT_AUTHOR_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 2048
_OPENING = (
    "Describe the operation you'd like to model. What is it, and what are the main things you can "
    "control and the outcomes you care about?"
)


@dataclass(frozen=True, slots=True)
class Session:
    """The elicitation dialogue state: the transcript, the running brief, and the pending question.

    ``question`` is the prompt to show the user next; it is ``None`` exactly when ``ready`` is True
    (the brief is complete or the user forced it). ``transcript`` is ``(role, text)`` turns.
    """

    transcript: tuple[tuple[str, str], ...]
    brief: WorldBrief
    question: str | None
    ready: bool


def start_session() -> Session:
    """Begin a session with an empty brief and the opening question (no LLM call yet)."""
    return Session(
        transcript=(("assistant", _OPENING),),
        brief=WorldBrief(),
        question=_OPENING,
        ready=False,
    )


def respond(elicitor: Elicitor, session: Session, message: str) -> Session:
    """Advance the dialogue with the user's ``message``: update the brief and ask the next question.

    Delegates the brief update + question choice to the elicitor (the author model). ``ready`` is
    set when the elicitor returns no further question.
    """
    transcript = (*session.transcript, ("user", message))
    brief, question = elicitor.advance(transcript, session.brief)
    closing = question if question is not None else "Brief complete — ready to generate."
    return Session(
        transcript=(*transcript, ("assistant", closing)),
        brief=brief,
        question=question,
        ready=question is None,
    )


def force_ready(session: Session) -> Session:
    """Hand off now on the user's say-so ("go"), even if the checklist is not fully satisfied."""
    return replace(session, question=None, ready=True)


class _BriefModel(BaseModel):
    """The structured brief the elicitor maintains (the pydantic boundary)."""

    domain: str = Field(default="", description="The operation in one or two sentences.")
    variables: list[str] = Field(
        default_factory=list,
        description="Lines like 'price (controllable): shelf price'. Role in parentheses.",
    )
    relationships: list[str] = Field(default_factory=list, description="Relationship hint lines.")
    regimes: str = Field(default="", description="Seasonality / regime switches, or '' if none.")
    hidden: str = Field(default="", description="Suspected hidden common causes, or '' if none.")
    objective: str = Field(default="", description="Control objective, or '' if none.")

    def to_brief(self) -> WorldBrief:
        """Convert to the frozen core ``WorldBrief``."""
        return WorldBrief(
            domain=self.domain,
            variables=tuple(self.variables),
            relationships=tuple(self.relationships),
            regimes=self.regimes,
            hidden=self.hidden,
            objective=self.objective,
        )


class _Step(BaseModel):
    """One elicitor turn: the merged brief plus the next question (or ready)."""

    brief: _BriefModel
    question: str | None = Field(
        default=None, description="The single most useful next question, or null when ready."
    )
    ready: bool = Field(
        default=False, description="True when the brief is complete enough to author."
    )


_SYSTEM = """\
You help a user SPECIFY a small, fictional causal OPERATION for a causal-discovery benchmark — you
do not grade it. Maintain a structured brief, merging everything the user has said so far.

Ask the user ONE concise clarifying question at a time, for the most important missing or ambiguous
item, in this rough priority: the operation/domain; its key variables and each one's role
(controllable lever / observable / disturbance / outcome KPI); what drives what; any regimes or
seasonality; suspected hidden common causes; and (optionally) the objective to optimize.

Infer sensible defaults and do NOT over-ask — aim for a usable brief in a handful of turns. Set
ready=true and omit the question once the brief has a domain, at least three variables with roles,
and at least one relationship. Return the full merged brief every turn."""


class ClaudeElicitor:
    """Drives elicitation via an injected ``instructor`` Claude client (the author model family)."""

    def __init__(self, client: instructor.Instructor, model: str = DEFAULT_AUTHOR_MODEL) -> None:
        """Store the client and model id (the client is constructed at the edge)."""
        self._client = client
        self._model = model

    def advance(
        self,
        transcript: Sequence[tuple[str, str]],
        brief: WorldBrief,  # noqa: ARG002 - the LLM re-derives the brief from the full transcript
    ) -> tuple[WorldBrief, str | None]:
        """Return ``(updated brief, next question)`` — question ``None`` when the brief is ready."""
        messages: Any = [{"role": "system", "content": _SYSTEM}, *self._dialogue(transcript)]
        step: _Step = self._client.chat.completions.create(
            model=self._model, max_tokens=_MAX_TOKENS, response_model=_Step, messages=messages
        )
        updated = step.brief.to_brief()
        # Trust the model's readiness only when the checklist actually agrees, or it asked nothing.
        ready = step.ready and is_complete(updated)
        question = None if ready else (step.question or _OPENING)
        return updated, question

    @staticmethod
    def _dialogue(transcript: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
        """Map the ``(role, text)`` transcript to provider chat messages."""
        return [{"role": role, "content": text} for role, text in transcript]


def build_claude_elicitor(
    model: str = DEFAULT_AUTHOR_MODEL, *, api_key: str | None = None
) -> ClaudeElicitor:  # pragma: no cover - real provider wiring, exercised only in live runs
    """Construct a live Claude elicitor; needs the ``llm`` extra and an Anthropic API key in env."""
    import instructor  # noqa: PLC0415 - lazy: the provider SDK is an optional `llm` extra
    from anthropic import Anthropic  # noqa: PLC0415

    return ClaudeElicitor(instructor.from_anthropic(Anthropic(api_key=api_key)), model)
