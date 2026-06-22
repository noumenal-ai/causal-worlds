"""Tests for the author->gate->admit loop (driven by fakes, no API key)."""

import pytest

from causal_worlds import worlds
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.fakes import FakeAuthor, FakeJudge
from causal_worlds.gates import GateReport
from causal_worlds.generate import NotAdmittedError, _feedback, generate, generate_many
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec

_FAST = InterventionalCiDiscoverer(n=4000)


def _bad_spec() -> WorldSpec:
    # an outcome but no controllable -> fails T1 (RoleError).
    return WorldSpec(
        variables=(Variable("a", Role.OBSERVABLE), Variable("b", Role.OUTCOME)),
        mechanisms=(Mechanism("b", (Term("a", 1.0),)),),
    )


def test_admits_a_good_world_first_try():
    author = FakeAuthor([worlds.get("ecommerce")])
    world = generate("a webshop", author=author, discoverer=_FAST, seed=7)
    assert world.report.admitted
    assert world.attempts == 1
    assert author.calls[0] == ("a webshop", None)  # first ask has no feedback


def test_re_authors_with_feedback_after_a_failed_gate():
    author = FakeAuthor([_bad_spec(), worlds.get("ecommerce")])
    world = generate("a webshop", author=author, discoverer=_FAST, seed=7)
    assert world.attempts == 2
    second_prompt, second_feedback = author.calls[1]
    assert second_prompt == "a webshop"
    assert second_feedback is not None
    assert second_feedback.startswith("Your previous attempt failed: T1")


def test_raises_when_never_admitted():
    author = FakeAuthor([_bad_spec()])
    with pytest.raises(NotAdmittedError) as exc:
        generate("x", author=author, discoverer=_FAST, seed=7, max_attempts=2)
    assert exc.value.attempts == 2
    assert exc.value.last is not None


def test_raises_with_no_attempts_budget():
    author = FakeAuthor([worlds.get("ecommerce")])
    with pytest.raises(NotAdmittedError) as exc:
        generate("x", author=author, discoverer=_FAST, max_attempts=0)
    assert exc.value.last is None


def test_feedback_falls_back_for_an_unknown_reason():
    fallback = _feedback(GateReport(admitted=False, reason="weird", null_shd=0.0, grade=None))
    assert "weird" in fallback
    cliche = _feedback(
        GateReport(admitted=False, reason="T4 cliché: ...", null_shd=0.0, grade=None)
    )
    assert "hidden" in cliche


def test_t4_runs_when_a_judge_is_supplied():
    author = FakeAuthor([worlds.get("coffee")])
    world = generate("a coffee chain", author=author, discoverer=_FAST, judge=FakeJudge(), seed=7)
    assert world.report.admitted
    assert world.report.difficulty == 1.0  # judge guessed nothing -> maximally anti-cliché


def test_generate_many_returns_one_outcome_per_prompt():
    author = FakeAuthor([worlds.get("ecommerce")])
    outcomes = generate_many(["a", "b"], author=author, discoverer=_FAST, judge=FakeJudge(), seed=7)
    assert [o.prompt for o in outcomes] == ["a", "b"]
    assert all(o.world is not None for o in outcomes)


def test_generate_many_records_failures_without_raising():
    author = FakeAuthor([_bad_spec()])  # never admittable
    outcomes = generate_many(["x"], author=author, discoverer=_FAST, seed=7, max_attempts=1)
    assert outcomes[0].world is None
    assert "not admitted" in outcomes[0].reason
