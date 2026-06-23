"""The ``WorldBrief`` — structured *intent* for a world, the artifact elicitation produces.

A one-shot prose prompt is underspecified. Conversational elicitation (:mod:`causal_worlds.elicit`)
accumulates a ``WorldBrief`` instead: the human-facing statement of *what* the operation is — its
entities and roles, what drives what, any regimes, suspected hidden common causes, and (for the
control track) an objective. The author then compiles a brief into the executable ``WorldSpec``.

Brief vs spec is the same split as everywhere else: the brief is intent (human), the spec is the
machine model. This module is the pure core — the completeness checklist and prose rendering, no IO.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorldBrief:
    """Structured intent for one fictional operation, filled incrementally during elicitation.

    ``variables`` are human lines like ``"price (controllable): shelf price"``; ``relationships``
    are hints like ``"price -> demand: higher price lowers demand"``. ``regimes``, ``hidden``, and
    ``objective`` are free text (``""`` = not yet specified / none).
    """

    domain: str = ""
    variables: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    regimes: str = ""
    hidden: str = ""
    objective: str = ""


_MIN_VARIABLES = 3  # a brief needs at least this many named entities to be worth authoring
_REQUIRED = ("domain", "variables", "relationships")  # the dimensions a complete brief must cover


def missing_fields(brief: WorldBrief) -> tuple[str, ...]:
    """The required dimensions a brief still lacks (empty when the brief is complete).

    ``regimes``, ``hidden``, and ``objective`` are optional and never reported here — they are
    prompted once but a world need not have them.
    """
    gaps: list[str] = []
    if not brief.domain.strip():
        gaps.append("domain")
    if len(brief.variables) < _MIN_VARIABLES:
        gaps.append("variables")
    if not brief.relationships:
        gaps.append("relationships")
    return tuple(gaps)


def is_complete(brief: WorldBrief) -> bool:
    """True once the brief covers every required dimension (domain, variables, relationships)."""
    return not missing_fields(brief)


def render(brief: WorldBrief) -> str:
    """Render a brief into the prose prompt the author consumes (also shown to the user)."""
    variables = "\n".join(f"- {line}" for line in brief.variables) or "- (none yet)"
    relationships = "\n".join(f"- {line}" for line in brief.relationships) or "- (none yet)"
    return (
        f"Operation: {brief.domain or '(unspecified)'}\n\n"
        f"Variables:\n{variables}\n\n"
        f"Known or suspected relationships:\n{relationships}\n\n"
        f"Regimes / seasonality: {brief.regimes or 'none specified'}\n"
        f"Suspected hidden common causes: {brief.hidden or 'none specified'}\n"
        f"Objective: {brief.objective or 'none specified'}"
    )
