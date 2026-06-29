"""Tests for `is_control_identifiable` (#27) — the observational backdoor-adjustment check.

The four canonical structures: a hidden lever↔outcome confounder (non-identifiable), an observed
confounder (adjust for it), a mediator (must not adjust), and an exogenous lever (no backdoor).
"""

from causal_worlds import default_objective, is_control_identifiable, worlds
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec

C, OUT, D = Role.CONTROLLABLE, Role.OUTCOME, Role.DISTURBANCE


def _world(variables: tuple[Variable, ...], mechanisms: tuple[Mechanism, ...]) -> WorldSpec:
    return WorldSpec(variables=variables, mechanisms=mechanisms)


def test_hidden_lever_outcome_confounder_is_not_identifiable() -> None:
    spec = _world(
        (Variable("H", D, hidden=True), Variable("L", C), Variable("Y", OUT)),
        (Mechanism("L", (Term("H", 0.9),)), Mechanism("Y", (Term("L", 0.5), Term("H", 1.2)))),
    )
    result = is_control_identifiable(spec, default_objective(spec))["L"]
    assert result.identifiable is False
    assert result.adjustment_set is None


def test_observed_confounder_is_identifiable_by_adjusting_for_it() -> None:
    spec = _world(
        (Variable("Ho", D), Variable("L", C), Variable("Y", OUT)),
        (Mechanism("L", (Term("Ho", 0.9),)), Mechanism("Y", (Term("L", 0.5), Term("Ho", 1.2)))),
    )
    result = is_control_identifiable(spec, default_objective(spec))["L"]
    assert result.identifiable is True
    assert result.adjustment_set is not None
    assert "Ho" in result.adjustment_set  # the observed confounder must be in the backdoor set


def test_mediator_is_identifiable_without_adjusting_for_the_mediator() -> None:
    spec = _world(
        (Variable("L", C), Variable("M", OUT), Variable("Y", OUT)),
        (Mechanism("M", (Term("L", 0.8),)), Mechanism("Y", (Term("M", 0.8),))),
    )
    result = is_control_identifiable(spec, default_objective(spec))["L"]
    assert result.identifiable is True
    assert result.adjustment_set is not None
    assert "M" not in result.adjustment_set  # a mediator (descendant of L) must NOT be adjusted for


def test_exogenous_lever_is_identifiable() -> None:
    # coffee's `price` is a root with no incoming edges → no backdoor path to block.
    spec = worlds.get("coffee")
    result = is_control_identifiable(spec, default_objective(spec))
    assert result["price"].identifiable is True
