"""Tests for the validity gate pipeline."""

from causal_worlds import answer_key, temporal_answer_key, worlds
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.fakes import FakeJudge, FakeTemporalDiscoverer
from causal_worlds.gates import run_gates
from causal_worlds.protocols import Edges, Substrate
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec

_FAST = InterventionalCiDiscoverer(n=4000)


class _BlindDiscoverer:
    """Recovers nothing — used to prove admission no longer depends on the grader."""

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:  # noqa: ARG002
        return frozenset()


class _RoleGuesser:
    """Guesses nothing from real names but recovers the truth once names are anonymized.

    Isolates the roles-only T4 gate: the leak is purely the role labels, not the names.
    """

    def prior_edges(self, spec, *, blind: bool = False):
        if blind:
            return frozenset()
        anonymized = all(v.name.startswith("X") for v in spec.variables)
        return answer_key(spec).edges if anonymized else frozenset()

    def faithfulness(self, prose: str, spec) -> float:  # noqa: ARG002
        return 1.0


def test_coffee_admitted_and_confounder_dropped():
    report = run_gates(worlds.get("coffee"), discoverer=_FAST, seed=7)
    assert report.admitted
    assert report.grade is not None
    assert report.grade.confounded_reported == 0
    assert report.difficulty is None  # no judge -> T4 did not run


def test_t4_admits_with_a_difficulty_score():
    # a judge that guesses nothing from priors -> maximally anti-cliché, fully faithful.
    report = run_gates(
        worlds.get("coffee"), discoverer=_FAST, seed=7, judge=FakeJudge(), prose="a coffee chain"
    )
    assert report.admitted
    assert report.difficulty == 1.0
    assert report.faithfulness == 1.0


def test_t4_rejects_a_cliche_world():
    # a judge that recovers the truth from priors alone -> guessable -> rejected.
    judge = FakeJudge(prior=answer_key(worlds.get("coffee")).edges)
    report = run_gates(
        worlds.get("coffee"), discoverer=_FAST, seed=7, judge=judge, prose="a coffee chain"
    )
    assert not report.admitted
    assert "T4 cliché" in report.reason
    assert report.difficulty == 0.0


def test_t4_rejects_a_half_guessable_world_under_the_strict_gate():
    # the named prior recovers 3 of 5 truth edges -> F1 ~0.67 >= 0.5 -> rejected (strict gate).
    truth = answer_key(worlds.get("ecommerce")).edges
    half = frozenset(list(truth)[: max(1, len(truth) // 2 + 1)])
    judge = FakeJudge(prior=half)
    report = run_gates(
        worlds.get("ecommerce"), discoverer=_FAST, seed=7, judge=judge, prose="an ecommerce store"
    )
    assert not report.admitted
    assert "T4 cliché" in report.reason


def test_t4_rejects_a_structural_cliche_via_the_blind_control():
    # names give nothing (prior empty), but the BLIND prior nails the truth -> structural cliché.
    truth = answer_key(worlds.get("coffee")).edges
    judge = FakeJudge(prior=frozenset(), blind_prior=truth)
    report = run_gates(
        worlds.get("coffee"), discoverer=_FAST, seed=7, judge=judge, prose="a coffee chain"
    )
    assert not report.admitted
    assert "structural cliché" in report.reason


def test_t4_rejects_an_unfaithful_spec():
    report = run_gates(
        worlds.get("coffee"),
        discoverer=_FAST,
        seed=7,
        judge=FakeJudge(score=0.3),
        prose="something else entirely",
    )
    assert not report.admitted
    assert "T4 unfaithful" in report.reason
    assert report.faithfulness == 0.3


def test_ecommerce_admitted():
    assert run_gates(worlds.get("ecommerce"), discoverer=_FAST, seed=7).admitted


def _temporal_world() -> WorldSpec:
    """A lagged world: lever ->(lag1) mid -> kpi. Temporal, so it routes to the temporal gate."""
    return WorldSpec(
        variables=(
            Variable("lever", Role.CONTROLLABLE),
            Variable("mid", Role.OBSERVABLE),
            Variable("kpi", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism("mid", (Term("lever", 0.8, lag=1),)),
            Mechanism("kpi", (Term("mid", 0.9),)),
        ),
    )


def test_temporal_world_also_runs_anti_cliche():
    # Temporal worlds must clear T4 too (the gap that could invalidate the name-dependence eval).
    spec = _temporal_world()
    recoverable = FakeTemporalDiscoverer(temporal_answer_key(spec))  # T3 passes (F1 1.0)
    # cliché: a judge that recovers the summary graph from names -> rejected at T4.
    cliche = run_gates(
        spec,
        temporal_discoverer=recoverable,
        judge=FakeJudge(prior=answer_key(spec).edges),
        prose="a lagged operation",
        seed=7,
    )
    assert not cliche.admitted
    assert "cliché" in cliche.reason
    # anti-cliché: judge guesses nothing -> admitted, carrying both temporal_grade and difficulty.
    clean = run_gates(
        spec, temporal_discoverer=recoverable, judge=FakeJudge(), prose="a lagged operation", seed=7
    )
    assert clean.admitted
    assert clean.temporal_grade is not None
    assert clean.difficulty == 1.0


def test_t4_rejects_a_role_cliche():
    # names give nothing, but the roles-only (name-anonymized) prior nails it -> role cliché (#13).
    report = run_gates(
        worlds.get("coffee"), discoverer=_FAST, seed=7, judge=_RoleGuesser(), prose="a coffee chain"
    )
    assert not report.admitted
    assert "role cliché" in report.reason


def test_admission_is_grader_independent():
    # The decisive decoupling test: a world faithful by construction is admitted even when the
    # supplied discoverer recovers nothing. Admission is now a property of the SCM, not the grader.
    report = run_gates(worlds.get("coffee"), discoverer=_BlindDiscoverer(), seed=7)
    assert report.admitted
    assert report.grade is not None
    assert report.grade.directed_shd > 0  # the blind grader failed — but the world still admits


def test_invalid_spec_rejected_at_t1():
    # an outcome but no controllable -> T1 RoleError -> rejected, not graded
    spec = WorldSpec(
        variables=(Variable("a", Role.OBSERVABLE), Variable("b", Role.OUTCOME)),
        mechanisms=(Mechanism("b", (Term("a", 1.0),)),),
    )
    report = run_gates(spec, discoverer=_FAST, seed=1)
    assert not report.admitted
    assert "T1" in report.reason
    assert report.grade is None
