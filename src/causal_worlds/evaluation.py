"""Score a discoverer's recovered structure against a world's ground-truth answer-key.

Pure set-math over directed edge sets — no IO, no numpy. ``directed_shd`` and ``f1`` measure
structure recovery; ``confounded_reported`` flags the key failure mode: claiming a *causal* edge for
a pair only **confounded** by a hidden cause (only interventions can tell them apart).

The module is named ``evaluation`` rather than ``eval`` to avoid shadowing the builtin.
"""

from dataclasses import dataclass

from causal_worlds.protocols import Edges, TemporalEdges
from causal_worlds.schema import AnswerKey


@dataclass(frozen=True, slots=True)
class Report:
    """The outcome of grading a recovered causal graph against the answer-key."""

    directed_shd: int
    skeleton_shd: int
    f1: float
    n_truth: int
    n_recovered: int
    confounded_reported: int


def directed_shd(recovered: Edges, truth: Edges) -> int:
    """Directed structural Hamming distance: missing + extra + reversed (a reversal counts once)."""
    missing = sum(
        1 for edge in truth if edge not in recovered and (edge[1], edge[0]) not in recovered
    )
    extra = sum(1 for edge in recovered if edge not in truth and (edge[1], edge[0]) not in truth)
    flipped = sum(1 for edge in recovered if edge not in truth and (edge[1], edge[0]) in truth)
    return missing + extra + flipped


def skeleton_shd(recovered: Edges, truth: Edges) -> int:
    """Undirected structural Hamming distance — adjacency errors, ignoring edge direction."""
    rec = {frozenset(edge) for edge in recovered}
    tru = {frozenset(edge) for edge in truth}
    return len(rec ^ tru)


def f1(recovered: Edges, truth: Edges) -> float:
    """F1 over directed edges — the harmonic mean of precision and recall."""
    if not truth and not recovered:
        return 1.0
    hits = len(recovered & truth)
    if hits == 0:
        return 0.0
    precision = hits / len(recovered)
    recall = hits / len(truth)
    return 2 * precision * recall / (precision + recall)


def _reports_edge(recovered: Edges, pair: frozenset[str]) -> bool:
    """True if ``recovered`` contains either orientation of an unordered pair."""
    left, right = sorted(pair)
    return (left, right) in recovered or (right, left) in recovered


def score(recovered: Edges, key: AnswerKey) -> Report:
    """Grade a recovered directed edge set against a world's answer-key."""
    confounded_reported = sum(1 for pair in key.confounded if _reports_edge(recovered, pair))
    return Report(
        directed_shd=directed_shd(recovered, key.edges),
        skeleton_shd=skeleton_shd(recovered, key.edges),
        f1=f1(recovered, key.edges),
        n_truth=len(key.edges),
        n_recovered=len(recovered),
        confounded_reported=confounded_reported,
    )


@dataclass(frozen=True, slots=True)
class TemporalReport:
    """The outcome of grading recovered lagged edges against the temporal answer-key."""

    temporal_shd: int
    temporal_f1: float
    n_truth: int
    n_recovered: int


def _reversal(edge: tuple[str, str, int]) -> tuple[str, str, int] | None:
    """The reversed edge — only meaningful at lag 0 (a lagged edge's direction is fixed by time)."""
    src, dst, lag = edge
    return (dst, src, lag) if lag == 0 else None


def temporal_directed_shd(recovered: TemporalEdges, truth: TemporalEdges) -> int:
    """Directed SHD over ``(src, dst, lag)`` edges: missing + extra + reversed (lag-0 reversals)."""
    missing = sum(1 for e in truth if e not in recovered and _reversal(e) not in recovered)
    extra = sum(1 for e in recovered if e not in truth and _reversal(e) not in truth)
    flipped = sum(
        1
        for e in recovered
        if e not in truth and _reversal(e) is not None and _reversal(e) in truth
    )
    return missing + extra + flipped


def temporal_f1(recovered: TemporalEdges, truth: TemporalEdges) -> float:
    """F1 over exact ``(src, dst, lag)`` triples."""
    if not truth and not recovered:
        return 1.0
    hits = len(recovered & truth)
    if hits == 0:
        return 0.0
    precision = hits / len(recovered)
    recall = hits / len(truth)
    return 2 * precision * recall / (precision + recall)


def temporal_score(recovered: TemporalEdges, truth: TemporalEdges) -> TemporalReport:
    """Grade recovered lagged edges against the temporal answer-key."""
    return TemporalReport(
        temporal_shd=temporal_directed_shd(recovered, truth),
        temporal_f1=temporal_f1(recovered, truth),
        n_truth=len(truth),
        n_recovered=len(recovered),
    )
