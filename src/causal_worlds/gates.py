"""The validity gates and the gate pipeline.

The gates run cheapest-first and fail loud — a world is admitted only if every gate passes:

* **T1 static validity** — the spec is well-formed (acyclic, roles present, no dangling refs).
* **T2 sample-sanity** — a sample is finite and non-degenerate (no zero-variance column).
* **T3 non-triviality** — the reference grader beats a per-world random-graph null by a margin.
* **T4 anti-cliché** — *only when an independent judge + the originating prose are supplied*: the
  spec faithfully represents the prose, and it is not guessable from priors alone. T4 also records a
  ``difficulty`` score (how far the judge's prior is from the truth) — the anti-cliché axis.

T1-T3 need no LLM (the v0.1 engine). T4 is the v0.2 author's partner: the judge must be a different
model family than the author, so a world is never graded easy by the same brain that wrote it.
"""

from dataclasses import dataclass

import numpy as np

from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.evaluation import Report, directed_shd, f1, score
from causal_worlds.protocols import Discoverer, Judge
from causal_worlds.sample import Sample, build_substrate
from causal_worlds.schema import AnswerKey, SpecError, WorldSpec, answer_key, validate

_NONTRIVIAL_FRACTION = 0.5  # admit iff the grader's directed SHD <= this * the random-graph null
_NULL_REPS = 1000
_SANITY_N = 500
_STD_EPS = 1e-9
_FAITHFUL_MIN = 0.6  # T4: reject a spec the judge deems an unfaithful reading of the prose
_CLICHE_MAX_F1 = 0.9  # T4: reject a world the judge all but recovers from priors alone (a cliché)


@dataclass(frozen=True, slots=True)
class GateReport:
    """The outcome of running the validity gates on a world.

    ``difficulty`` and ``faithfulness`` are populated only when T4 ran (a judge + prose were given);
    ``difficulty`` is ``1 - F1(judge_prior, truth)`` — higher means harder to guess from priors.
    """

    admitted: bool
    reason: str
    null_shd: float
    grade: Report | None
    difficulty: float | None = None
    faithfulness: float | None = None


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


def _is_degenerate(sample: Sample) -> bool:
    """True if a sample is non-finite or has a zero-variance column (nothing to discover)."""
    return not bool(np.all(np.isfinite(sample.data))) or bool(
        np.any(sample.data.std(axis=0) < _STD_EPS)
    )


def run_gates(
    spec: WorldSpec,
    *,
    discoverer: Discoverer | None = None,
    seed: int = 0,
    judge: Judge | None = None,
    prose: str | None = None,
) -> GateReport:
    """Run the validity gates; admit the world only if all pass.

    Args:
        spec: The candidate world.
        discoverer: The reference grader (defaults to the interventional-CI discoverer).
        seed: Determines sampling and the random-null baseline.
        judge: An independent LLM judge; enables T4 (anti-cliché) when given with ``prose``.
        prose: The natural-language description the spec was authored from (for T4 faithfulness).

    Returns:
        A :class:`GateReport` with the admit decision, the failing gate's reason, the grade, and —
        when T4 ran — the anti-cliché ``difficulty`` and ``faithfulness`` scores.
    """
    try:
        validate(spec)
    except SpecError as exc:
        return GateReport(
            admitted=False, reason=f"T1 invalid spec: {exc}", null_shd=0.0, grade=None
        )

    substrate = build_substrate(spec)
    if _is_degenerate(substrate.sample(_SANITY_N, seed=seed)):
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
    if grade.directed_shd > _NONTRIVIAL_FRACTION * null_shd:
        return GateReport(
            admitted=False,
            reason=f"T3 not recoverable: SHD {grade.directed_shd} vs null {null_shd:.1f}",
            null_shd=null_shd,
            grade=grade,
        )

    if judge is None or prose is None:
        return GateReport(admitted=True, reason="admitted", null_shd=null_shd, grade=grade)
    admitted, reason, difficulty, faithfulness = _anti_cliche(spec, prose, judge, key)
    return GateReport(
        admitted=admitted,
        reason=reason,
        null_shd=null_shd,
        grade=grade,
        difficulty=difficulty,
        faithfulness=faithfulness,
    )


def _anti_cliche(
    spec: WorldSpec, prose: str, judge: Judge, key: AnswerKey
) -> tuple[bool, str, float | None, float]:
    """The T4 decision: ``(admitted, reason, difficulty, faithfulness)``.

    Rejects a spec the judge reads as unfaithful, or a world it all but recovers from priors alone
    (a cliché). Otherwise admits, carrying ``difficulty = 1 - F1(judge_prior, truth)``.
    """
    faithfulness = judge.faithfulness(prose, spec)
    if faithfulness < _FAITHFUL_MIN:
        return (
            False,
            f"T4 unfaithful to prompt: {faithfulness:.2f} < {_FAITHFUL_MIN}",
            None,
            faithfulness,
        )

    prior_f1 = f1(judge.prior_edges(spec), key.edges)
    difficulty = 1.0 - prior_f1
    if prior_f1 >= _CLICHE_MAX_F1:
        return (
            False,
            f"T4 cliché: recovered from priors (F1 {prior_f1:.2f})",
            difficulty,
            faithfulness,
        )
    return True, "admitted", difficulty, faithfulness
