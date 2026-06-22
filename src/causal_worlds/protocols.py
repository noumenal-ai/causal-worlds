"""Protocol contracts for the package's proven variation points (Dependency Inversion).

Concrete implementations live behind adapters; third-party types (``causal-learn``, ``gies``,
Gemini) never leak past an adapter. Depend on these Protocols; inject the concrete impl at the edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from causal_worlds.schema import WorldSpec

Edges = frozenset[tuple[str, str]]
"""A set of directed edges ``(src, dst)`` over variable names."""


@runtime_checkable
class Judge(Protocol):
    """An independent LLM judge — must be a different model family than the world's author."""

    def prior_edges(self, spec: WorldSpec) -> Edges:
        """Guess the causal edges from variable names + prose alone, with no data."""
        ...

    def faithfulness(self, prose: str, spec: WorldSpec) -> float:
        """Score in ``[0, 1]`` how faithfully ``spec`` represents ``prose``."""
        ...


@runtime_checkable
class Substrate(Protocol):
    """An executable world: sample observational and interventional data."""

    def sample(self, n: int, *, do: dict[str, float] | None = None) -> object:
        """Sample ``n`` rows; ``do`` intervenes by fixing the named variables."""
        ...


@runtime_checkable
class Discoverer(Protocol):
    """A causal-discovery grader — interventional-CI, not a stock GES/GIES call."""

    def recover(self, world: Substrate) -> Edges:
        """Recover directed edges from the world's observational + interventional data."""
        ...


@runtime_checkable
class Gate(Protocol):
    """One check in the author->gate->admit pipeline; gates compose, cheapest first."""

    def check(self, world: Substrate) -> bool:
        """Return ``True`` if the world passes this gate."""
        ...
