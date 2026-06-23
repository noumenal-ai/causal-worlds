"""Protocol contracts for the package's proven variation points (Dependency Inversion).

Concrete implementations live behind adapters; third-party types (``causal-learn``, ``gies``,
Gemini) never leak past an adapter. Depend on these Protocols; inject the concrete impl at the edge.

It keeps ``from __future__ import annotations`` deliberately — it annotates with types imported
only under ``TYPE_CHECKING`` (kept out of the runtime graph) and uses forward references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from causal_worlds.brief import WorldBrief
    from causal_worlds.sample import FloatArray, Sample
    from causal_worlds.schema import WorldSpec

Edges = frozenset[tuple[str, str]]
"""A set of directed edges ``(src, dst)`` over variable names."""

TemporalEdges = frozenset[tuple[str, str, int]]
"""Directed temporal edges ``(src, dst, lag)``; lag 0 is contemporaneous, >= 1 is lagged."""


@runtime_checkable
class Elicitor(Protocol):
    """Drives the clarify dialogue toward a complete ``WorldBrief`` (the author model, an adapter).

    Stateless across calls: it reads the whole transcript each turn and returns the updated brief
    plus the next question (``None`` when the brief is ready). The :class:`Session` holds the state.
    """

    def advance(
        self, transcript: Sequence[tuple[str, str]], brief: WorldBrief
    ) -> tuple[WorldBrief, str | None]:
        """Given the dialogue and the running brief, return ``(updated brief, next question)``.

        ``next question`` is ``None`` exactly when the elicitor judges the brief complete.
        """
        ...


@runtime_checkable
class Author(Protocol):
    """Manufactures a world spec from a natural-language description (an LLM, behind an adapter)."""

    def author(self, prompt: str, *, feedback: str | None = None) -> WorldSpec:
        """Author a :class:`WorldSpec` from prose; ``feedback`` re-asks after a failed gate."""
        ...


@runtime_checkable
class Judge(Protocol):
    """An independent LLM judge — a different model family than the world's author."""

    def prior_edges(self, spec: WorldSpec, *, blind: bool = False) -> Edges:
        """Guess the causal edges from priors alone (no data).

        ``blind`` hides names (anonymized) and roles — a control that should score at chance.
        """
        ...

    def faithfulness(self, prose: str, spec: WorldSpec) -> float:
        """Score in ``[0, 1]`` how faithfully ``spec`` represents ``prose``."""
        ...


@runtime_checkable
class Substrate(Protocol):
    """An executable world: sample observational and interventional data, deterministically."""

    @property
    def variables(self) -> tuple[str, ...]:
        """The observed variable names, in the column order of sampled data."""
        ...

    def sample(
        self, n: int, *, seed: int, do: Mapping[str, float | FloatArray] | None = None
    ) -> Sample:
        """Sample ``n`` rows; ``do`` fixes variables to constants or per-row arrays."""
        ...


@runtime_checkable
class Discoverer(Protocol):
    """A causal-discovery grader — interventional-CI, not a stock GES/GIES call."""

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:
        """Recover directed edges, driving its own observational + interventional sampling."""
        ...


@runtime_checkable
class TemporalDiscoverer(Protocol):
    """A time-series causal-discovery method — recovers lagged edges from temporal data."""

    def recover_temporal(self, substrate: Substrate, *, seed: int) -> TemporalEdges:
        """Recover directed lagged edges ``(src, dst, lag)`` from the substrate's time-series."""
        ...


@runtime_checkable
class Gate(Protocol):
    """One check in the author->gate->admit pipeline; gates compose, cheapest first."""

    def check(self, substrate: Substrate) -> bool:
        """Return ``True`` if the world passes this gate."""
        ...
