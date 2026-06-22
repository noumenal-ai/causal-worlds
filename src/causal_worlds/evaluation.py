"""Score a discoverer's recovered structure against a world's ground-truth answer-key.

Pure set-math over directed edge sets — no IO, no numpy. ``directed_shd`` and ``f1`` measure
structure recovery; ``confounded_reported`` flags the failure mode that matters most here: claiming a
*causal* edge for a pair only **confounded** by a hidden cause (only interventions tell them apart).

The module is named ``evaluation`` rather than ``eval`` to avoid shadowing the builtin.
"""

from dataclasses import dataclass

from causal_worlds.protocols import Edges
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
