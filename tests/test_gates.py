"""Tests for the validity gate pipeline."""

from causal_worlds import answer_key, worlds
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.fakes import FakeJudge
from causal_worlds.gates import run_gates
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec

_FAST = InterventionalCiDiscoverer(n=4000)


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
