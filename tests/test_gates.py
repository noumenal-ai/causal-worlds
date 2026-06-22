"""Tests for the validity gate pipeline."""

from causal_worlds import worlds
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.gates import run_gates
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec

_FAST = InterventionalCiDiscoverer(n=4000)


def test_coffee_admitted_and_confounder_dropped():
    report = run_gates(worlds.get("coffee"), discoverer=_FAST, seed=7)
    assert report.admitted
    assert report.grade is not None
    assert report.grade.confounded_reported == 0


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
