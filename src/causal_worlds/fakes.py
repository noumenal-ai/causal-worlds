"""Deterministic test doubles for the :class:`Author` and :class:`Judge` seams.

These let the whole author->gate->admit loop run with no API key — in unit tests, in CI, and as a
keyless demo. They are real implementations of the Protocols, just driven by canned data instead of
an LLM, so they exercise the same code paths the live adapters do.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from causal_worlds.brief import WorldBrief
from causal_worlds.protocols import Edges, Substrate, TemporalEdges
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
    blind_prior: Edges | None = None

    def prior_edges(self, spec: WorldSpec, *, blind: bool = False) -> Edges:  # noqa: ARG002
        """Return the canned prior (``blind_prior`` when blind and set, else ``prior``)."""
        if blind and self.blind_prior is not None:
            return self.blind_prior
        return self.prior

    def faithfulness(self, prose: str, spec: WorldSpec) -> float:  # noqa: ARG002
        """Return the canned faithfulness score."""
        return self.score


@dataclass(slots=True)
class FakeElicitor:
    """An :class:`Elicitor` that returns canned ``(brief, question)`` steps in order.

    Each call returns the next scripted step; a step with ``question=None`` signals the brief is
    ready. The last step repeats once the sequence is exhausted. Records the transcripts it saw.
    """

    steps: Sequence[tuple[WorldBrief, str | None]]
    calls: list[int] = field(default_factory=list)

    def advance(
        self,
        transcript: Sequence[tuple[str, str]],
        brief: WorldBrief,  # noqa: ARG002
    ) -> tuple[WorldBrief, str | None]:
        """Return the next scripted ``(brief, question)`` step, recording the transcript length."""
        index = min(len(self.calls), len(self.steps) - 1)
        self.calls.append(len(transcript))
        return self.steps[index]


@dataclass(slots=True)
class FakeTemporalDiscoverer:
    """A :class:`TemporalDiscoverer` returning canned lagged edges (for keyless temporal tests)."""

    edges: TemporalEdges = frozenset()

    def recover_temporal(self, substrate: Substrate, *, seed: int) -> TemporalEdges:  # noqa: ARG002
        """Return the canned lagged edges (independent of the substrate)."""
        return self.edges
