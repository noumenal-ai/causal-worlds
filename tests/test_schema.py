"""Tests for the world-spec IR, the static validation gate, and the derived answer-key."""

import pytest

from causal_worlds.schema import (
    AnswerKey,
    CyclicGraphError,
    DanglingReferenceError,
    DuplicateMechanismError,
    Mechanism,
    Role,
    RoleError,
    Term,
    Variable,
    answer_key,
    validate,
)
from causal_worlds.schema import WorldSpec as Spec


def _coffee():
    """A small anti-cliché world: hidden L confounds O and S; price->demand flips by regime R."""
    variables = (
        Variable("R", Role.DISTURBANCE),
        Variable("price", Role.CONTROLLABLE),
        Variable("L", Role.DISTURBANCE, hidden=True),
        Variable("foot", Role.OBSERVABLE),
        Variable("overtime", Role.OBSERVABLE),
        Variable("demand", Role.OBSERVABLE),
        Variable("sales", Role.OUTCOME),
    )
    mechanisms = (
        Mechanism("foot", (Term("L", 0.8),)),
        Mechanism("overtime", (Term("L", 0.8), Term("foot", 0.3))),
        Mechanism(
            "demand",
            terms=(Term("price", -1.0), Term("foot", 0.5)),
            regime="R",
            regime_terms=(Term("price", 1.0), Term("foot", 0.5)),
        ),
        Mechanism("sales", (Term("demand", 1.0), Term("foot", 0.4), Term("L", 0.6))),
    )
    return Spec(variables=variables, mechanisms=mechanisms)


def test_valid_spec_passes():
    validate(_coffee())  # should not raise


def test_answer_key_derives_observed_edges_and_confounding():
    key = answer_key(_coffee())
    assert isinstance(key, AnswerKey)
    # observed edges incl. price->demand, R->demand, foot->overtime, demand->sales
    assert ("price", "demand") in key.edges
    assert ("R", "demand") in key.edges
    assert ("demand", "sales") in key.edges
    # the hidden L confounds overtime & sales with no direct edge between them
    assert frozenset(("overtime", "sales")) in key.confounded
    # hidden L never appears as a node in the observed answer-key
    assert all("L" not in edge for edge in key.edges)


def test_dangling_reference_rejected():
    spec = Spec(
        variables=(Variable("price", Role.CONTROLLABLE), Variable("sales", Role.OUTCOME)),
        mechanisms=(Mechanism("sales", (Term("ghost", 1.0),)),),
    )
    with pytest.raises(DanglingReferenceError):
        validate(spec)


def test_cycle_rejected():
    spec = Spec(
        variables=(
            Variable("a", Role.CONTROLLABLE),
            Variable("b", Role.OBSERVABLE),
            Variable("c", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism("b", (Term("a", 1.0),)),
            Mechanism("c", (Term("b", 1.0),)),
            Mechanism("a", (Term("c", 1.0),)),
        ),
    )
    with pytest.raises(CyclicGraphError):
        validate(spec)


def test_duplicate_mechanism_rejected():
    spec = Spec(
        variables=(Variable("price", Role.CONTROLLABLE), Variable("sales", Role.OUTCOME)),
        mechanisms=(
            Mechanism("sales", (Term("price", 1.0),)),
            Mechanism("sales", (Term("price", 2.0),)),
        ),
    )
    with pytest.raises(DuplicateMechanismError):
        validate(spec)


def test_missing_controllable_rejected():
    spec = Spec(
        variables=(Variable("demand", Role.OBSERVABLE), Variable("sales", Role.OUTCOME)),
        mechanisms=(Mechanism("sales", (Term("demand", 1.0),)),),
    )
    with pytest.raises(RoleError):
        validate(spec)


def test_missing_outcome_rejected():
    spec = Spec(
        variables=(Variable("price", Role.CONTROLLABLE), Variable("demand", Role.OBSERVABLE)),
        mechanisms=(Mechanism("demand", (Term("price", 1.0),)),),
    )
    with pytest.raises(RoleError):
        validate(spec)
