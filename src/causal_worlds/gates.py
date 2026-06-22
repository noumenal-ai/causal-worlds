"""The validity gates and the gate pipeline.

v0.1 runs T1 (static validity), T2 (sample-sanity), and T3 (non-triviality vs a random-graph null,
normalized per world). T4 (anti-cliché, needing an independent LLM judge) lands with the author in
v0.2. A world is admitted only if every gate passes; on failure the gate's reason is reported (fail
loud — we never admit a world we can't justify).
"""

from dataclasses import dataclass

import numpy as np

from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.evaluation import Report, directed_shd, score
from causal_worlds.protocols import Discoverer
from causal_worlds.sample import build_substrate
from causal_worlds.schema import AnswerKey, SpecError, WorldSpec, answer_key, validate

_NONTRIVIAL_FRACTION = 0.5  # admit iff the grader's directed SHD <= this * the random-graph null
_NULL_REPS = 1000
_SANITY_N = 500
_STD_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class GateReport:
    """The outcome of running the validity gates on a world."""

    admitted: bool
    reason: str
    null_shd: float
    grade: Report | None


def _random_null_shd(key: AnswerKey, names: tuple[str, ...], seed: int, reps: int) -> float:
    """Mean directed SHD of random same-size graphs vs the answer-key — the chance floor."""
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in names for b in names if a != b]
    k = len(key.edges)
    if k == 0 or len(pairs) < k:
        return 0.0
    total = 0
    for _ in range(reps):
        chosen = rng.choice(len(pairs), size=k, replace=False)
        total += directed_shd(frozenset(pairs[int(i)] for i in chosen), key.edges)
    return total / reps


def run_gates(
    spec: WorldSpec, *, discoverer: Discoverer | None = None, seed: int = 0
) -> GateReport:
    """Run the validity gates; admit the world only if all pass.

    Args:
        spec: The candidate world.
        discoverer: The reference grader (defaults to the interventional-CI discoverer).
        seed: Determines sampling and the random-null baseline.

    Returns:
        A :class:`GateReport` with the admit decision, the failing gate's reason, and the grade.
    """
    try:
        validate(spec)
    except SpecError as exc:
        return GateReport(
            admitted=False, reason=f"T1 invalid spec: {exc}", null_shd=0.0, grade=None
        )

    substrate = build_substrate(spec)
    sample = substrate.sample(_SANITY_N, seed=seed)
    degenerate = not bool(np.all(np.isfinite(sample.data))) or bool(
        np.any(sample.data.std(axis=0) < _STD_EPS)
    )
    if degenerate:
        return GateReport(
            admitted=False,
            reason="T2 degenerate sample (non-finite or zero-variance column)",
            null_shd=0.0,
            grade=None,
        )

    key = answer_key(spec)
    if not key.edges:
        return GateReport(
            admitted=False, reason="T3 no causal structure to discover", null_shd=0.0, grade=None
        )

    grader = discoverer if discoverer is not None else InterventionalCiDiscoverer()
    grade = score(grader.recover(substrate, seed=seed), key)
    null_shd = _random_null_shd(key, substrate.variables, seed, _NULL_REPS)
    admitted = grade.directed_shd <= _NONTRIVIAL_FRACTION * null_shd
    reason = (
        "admitted"
        if admitted
        else f"T3 not recoverable: SHD {grade.directed_shd} vs null {null_shd:.1f}"
    )
    return GateReport(admitted=admitted, reason=reason, null_shd=null_shd, grade=grade)
