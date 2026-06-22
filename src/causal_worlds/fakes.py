"""Deterministic test doubles for the :class:`Author` and :class:`Judge` seams.

These let the whole author->gate->admit loop run with no API key — in unit tests, in CI, and as a
keyless demo. They are real implementations of the Protocols, just driven by canned data instead of
an LLM, so they exercise the same code paths the live adapters do.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from causal_worlds.protocols import Edges
from causal_worlds.schema import WorldSpec


@dataclass(slots=True)
class FakeAuthor:
    """An :class:`Author` that returns canned specs in order, recording each call.

    Pass one spec to always return it, or several to return a different spec per attempt (so a test
    can drive the bounded re-author loop). The last spec repeats once the sequence is exhausted.
    """

    specs: Sequence[WorldSpec]
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def author(self, prompt: str, *, feedback: str | None = None) -> WorldSpec:
        """Return the next canned spec, recording the (prompt, feedback) it was asked with."""
        index = min(len(self.calls), len(self.specs) - 1)
        self.calls.append((prompt, feedback))
        return self.specs[index]


@dataclass(slots=True)
class FakeJudge:
    """A :class:`Judge` returning canned prior edges and a canned faithfulness score.

    Default behaviour models a judge that guesses *nothing* from prose alone (maximally anti-cliché)
    and deems the spec fully faithful — the simple admit case. Override either to test the T4 gate.
    """

    prior: Edges = frozenset()
    score: float = 1.0

    def prior_edges(self, spec: WorldSpec) -> Edges:  # noqa: ARG002
        """Return the canned prior edges (independent of the spec)."""
        return self.prior

    def faithfulness(self, prose: str, spec: WorldSpec) -> float:  # noqa: ARG002
        """Return the canned faithfulness score."""
        return self.score
