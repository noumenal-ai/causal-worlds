"""The validity gates and the gate pipeline.

The gates run cheapest-first and fail loud — a world is admitted only if every gate passes:

* **T1 static validity** — the spec is well-formed (acyclic, roles present, no dangling refs).
* **T2 sample-sanity** — a sample is finite and non-degenerate (no zero-variance column).
* **T3 faithfulness (grader-independent)** — the declared SCM is *faithful by construction* (every
  edge induces a detectable partial correlation; regimes genuinely modulate) and structurally
  non-trivial. Computed in closed form from the spec — no discovery method is run, so admission
  privileges none. The reference grader's score is still recorded, but only as an observation.
* **T4 anti-cliché** — *only when an independent judge + the originating prose are supplied*: the
  spec faithfully represents the prose, and it is not guessable from priors alone. T4 also records a
  ``difficulty`` score (how far the judge's prior is from the truth) — the anti-cliché axis.

T1-T3 need no LLM (the v0.1 engine). T4 is the v0.2 author's partner: the judge must be a different
model family than the author, so a world is never graded easy by the same brain that wrote it.
"""

from dataclasses import dataclass

import numpy as np

from causal_worlds.admission import check_faithfulness, is_nontrivial
from causal_worlds.anonymize import anonymize_spec
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.evaluation import Report, TemporalReport, directed_shd, f1, score, temporal_score
from causal_worlds.protocols import Discoverer, Judge, Substrate, TemporalDiscoverer
from causal_worlds.sample import Sample, build_substrate
from causal_worlds.schema import (
    AnswerKey,
    SpecError,
    WorldSpec,
    answer_key,
    temporal_answer_key,
    validate,
)
from causal_worlds.temporal_baselines import PcmciPlusDiscoverer

_NULL_REPS = 1000
_SANITY_N = 500
_STD_EPS = 1e-9
_FAITHFUL_MIN = 0.6  # T4: reject a spec the judge deems an unfaithful reading of the prose
_CLICHE_MAX_F1 = 0.5  # T4: admit only if the named-prior guess recovers < half (difficulty >= 0.5)
_ROLES_ONLY_MAX_F1 = (
    0.4  # T4: the name-blind (roles-kept) prior must not recover it from roles alone
)
_BLIND_MAX_F1 = 0.35  # T4: the name+role-blind prior must sit near the chance floor (a control)
_TEMPORAL_F1_MIN = (
    0.5  # T3 (temporal): admit iff a TS reference recovers lagged edges above this F1
)


@dataclass(frozen=True, slots=True)
class GateReport:
    """The outcome of running the validity gates on a world.

    ``difficulty`` and ``faithfulness`` are populated only when T4 ran (a judge + prose were given);
    ``difficulty`` is ``1 - F1(judge_prior, truth)`` — higher means harder to guess from priors.
    ``temporal_grade`` replaces ``grade`` for temporal worlds (scored on lagged edges).
    """

    admitted: bool
    reason: str
    null_shd: float
    grade: Report | None
    difficulty: float | None = None
    faithfulness: float | None = None
    temporal_grade: TemporalReport | None = None


def _is_temporal(spec: WorldSpec) -> bool:
    """True if any term carries a lag (the world has temporal structure)."""
    return any(
        term.lag > 0
        for mechanism in spec.mechanisms
        for term in (*mechanism.terms, *(mechanism.regime_terms or ()))
    )


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


def run_gates(  # noqa: PLR0911, PLR0913 — one return per gate outcome; keyword-only knobs
    spec: WorldSpec,
    *,
    discoverer: Discoverer | None = None,
    seed: int = 0,
    judge: Judge | None = None,
    prose: str | None = None,
    temporal_discoverer: TemporalDiscoverer | None = None,
) -> GateReport:
    """Run the validity gates; admit the world only if all pass.

    Args:
        spec: The candidate world.
        discoverer: The reference grader (defaults to the interventional-CI discoverer).
        seed: Determines sampling and the random-null baseline.
        judge: An independent LLM judge; enables T4 (anti-cliché) when given with ``prose``.
        prose: The natural-language description the spec was authored from (for T4 faithfulness).
        temporal_discoverer: The reference TS grader for temporal worlds (defaults to PCMCI+).

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

    if _is_temporal(spec):
        return _temporal_gates(spec, substrate, temporal_discoverer, seed, judge=judge, prose=prose)

    key = answer_key(spec)
    if not key.edges:
        return GateReport(
            admitted=False, reason="T3 no causal structure to discover", null_shd=0.0, grade=None
        )

    # T3 is now GRADER-INDEPENDENT: admit on a closed-form property of the declared SCM (every edge
    # faithful, structure non-trivial), never on whether the reference grader recovers the world.
    if not is_nontrivial(spec):
        return GateReport(
            admitted=False, reason="T3 trivial: (near-)complete graph", null_shd=0.0, grade=None
        )
    faith = check_faithfulness(spec)
    if not faith.faithful:
        return GateReport(admitted=False, reason=f"T3 {faith.reason}", null_shd=0.0, grade=None)

    # The reference grader is still run, but only to *report* its score on this independently-
    # admitted world, not to decide admission. The random-null is recorded for context.
    grader = discoverer if discoverer is not None else InterventionalCiDiscoverer()
    grade = score(grader.recover(substrate, seed=seed), key)
    null_shd = _random_null_shd(key, substrate.variables, seed, _NULL_REPS)

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


def _temporal_gates(  # noqa: PLR0913 — keyword-only judge/prose mirror run_gates' T4 inputs
    spec: WorldSpec,
    substrate: Substrate,
    temporal_discoverer: TemporalDiscoverer | None,
    seed: int,
    *,
    judge: Judge | None = None,
    prose: str | None = None,
) -> GateReport:
    """Temporal T3 (a TS reference recovers the lagged structure) + T4 anti-cliché when judged.

    T4 runs on the **summary graph** (``answer_key`` collapses lags), so a temporal world is
    admitted only if it is recoverable *and* not guessable from names/roles — the same anti-cliché
    bar the cross-sectional path enforces. (Temporal grader-independent faithfulness is future work;
    PCMCI+ recoverability is the temporal recoverability proxy here.)
    """
    truth = temporal_answer_key(spec)
    if not truth:
        return GateReport(
            admitted=False, reason="T3 no temporal structure to discover", null_shd=0.0, grade=None
        )
    grader = temporal_discoverer if temporal_discoverer is not None else PcmciPlusDiscoverer()
    report = temporal_score(grader.recover_temporal(substrate, seed=seed), truth)
    if report.temporal_f1 < _TEMPORAL_F1_MIN:
        return GateReport(
            admitted=False,
            reason=f"T3 temporal structure not recoverable: F1 {report.temporal_f1:.2f}",
            null_shd=0.0,
            grade=None,
            temporal_grade=report,
        )
    if judge is None or prose is None:
        return GateReport(
            admitted=True, reason="admitted", null_shd=0.0, grade=None, temporal_grade=report
        )
    admitted, reason, difficulty, faithfulness = _anti_cliche(spec, prose, judge, answer_key(spec))
    return GateReport(
        admitted=admitted,
        reason=reason,
        null_shd=0.0,
        grade=None,
        temporal_grade=report,
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
            f"T4 cliché: names+roles recover it (prior F1 {prior_f1:.2f} >= {_CLICHE_MAX_F1})",
            difficulty,
            faithfulness,
        )
    # Roles-only control: with names anonymized but roles kept, the prior must not recover it from
    # role conventions alone (controllable -> outcome). Closes the role-label leak (#13).
    anon, _ = anonymize_spec(spec)
    roles_f1 = f1(judge.prior_edges(anon), answer_key(anon).edges)
    if roles_f1 >= _ROLES_ONLY_MAX_F1:
        return (
            False,
            f"T4 role cliché: roles alone recover it (F1 {roles_f1:.2f} >= {_ROLES_ONLY_MAX_F1})",
            difficulty,
            faithfulness,
        )
    # Structural control (Caliper): with names AND roles hidden, the prior must sit near chance —
    # otherwise the structure itself is a cliché (a canonical chain a guesser nails blind).
    blind_f1 = f1(judge.prior_edges(spec, blind=True), key.edges)
    if blind_f1 >= _BLIND_MAX_F1:
        return (
            False,
            f"T4 structural cliché: blind prior recovers it (F1 {blind_f1:.2f} >= {_BLIND_MAX_F1})",
            difficulty,
            faithfulness,
        )
    return True, "admitted", difficulty, faithfulness
