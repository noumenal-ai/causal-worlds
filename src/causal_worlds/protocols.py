"""Protocol contracts for the package's proven variation points (Dependency Inversion).

Concrete implementations live behind adapters; third-party types (``causal-learn``, ``gies``,
Gemini) never leak past an adapter. Depend on these Protocols; inject the concrete impl at the edge.

It keeps ``from __future__ import annotations`` deliberately — it annotates with types imported
only under ``TYPE_CHECKING`` (kept out of the runtime graph) and uses forward references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from causal_worlds.sample import Sample
    from causal_worlds.schema import WorldSpec

Edges = frozenset[tuple[str, str]]
"""A set of directed edges ``(src, dst)`` over variable names."""


@runtime_checkable
class Judge(Protocol):
    """An independent LLM judge — a different model family than the world's author."""

    def prior_edges(self, spec: WorldSpec) -> Edges:
        """Guess the causal edges from variable names + prose alone, with no data."""
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

    def sample(self, n: int, *, seed: int, do: Mapping[str, float] | None = None) -> Sample:
        """Sample ``n`` rows; ``do`` fixes the named variables (an intervention)."""
        ...


@runtime_checkable
class Discoverer(Protocol):
    """A causal-discovery grader — interventional-CI, not a stock GES/GIES call."""

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:
        """Recover directed edges, driving its own observational + interventional sampling."""
        ...


@runtime_checkable
class Gate(Protocol):
    """One check in the author->gate->admit pipeline; gates compose, cheapest first."""

    def check(self, substrate: Substrate) -> bool:
        """Return ``True`` if the world passes this gate."""
        ...
