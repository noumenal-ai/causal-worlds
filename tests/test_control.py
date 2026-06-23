"""Tests for the Stage-2 control answer-key (optimal policy + regret)."""

import pytest

from causal_worlds.control import (
    ControlObjective,
    default_objective,
    grade_control,
    grade_controller,
    lever_effects,
    optimal_policy,
    regime_optimal_policy,
    regret_under_perturbation,
)
from causal_worlds.protocols import Controller, Substrate
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


def _chain() -> WorldSpec:
    """price -> demand (0.5) -> sales (2.0): total effect price->sales = 1.0."""
    return WorldSpec(
        variables=(
            Variable("price", Role.CONTROLLABLE),
            Variable("demand", Role.OBSERVABLE),
            Variable("sales", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism("demand", (Term("price", 0.5),)),
            Mechanism("sales", (Term("demand", 2.0),)),
        ),
    )


def _sign_flip() -> WorldSpec:
    """price -> sales, +1 in one regime and -1 in the other: marginal effect cancels to 0."""
    return WorldSpec(
        variables=(
            Variable("R", Role.DISTURBANCE),
            Variable("price", Role.CONTROLLABLE),
            Variable("sales", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism(
                "sales",
                terms=(Term("price", 1.0),),
                regime="R",
                regime_terms=(Term("price", -1.0),),
            ),
        ),
    )


class _FixedController:
    """A controller that always plays a canned policy (a control test-taker double)."""

    def __init__(self, policy):
        self._policy = policy

    def control(self, substrate: Substrate, objective: ControlObjective, *, seed: int):  # noqa: ARG002
        return self._policy


def test_default_objective_reads_roles():
    obj = default_objective(_chain())
    assert obj.levers == ("price",)
    assert obj.outcome == "sales"


def test_default_objective_requires_a_lever_and_an_outcome():
    spec = WorldSpec(
        variables=(Variable("a", Role.OBSERVABLE), Variable("b", Role.OUTCOME)),
        mechanisms=(Mechanism("b", (Term("a", 1.0),)),),
    )
    with pytest.raises(ValueError, match="controllable lever"):
        default_objective(spec)


def test_lever_effects_is_the_path_sum():
    spec = _chain()
    assert lever_effects(spec, default_objective(spec))["price"] == pytest.approx(1.0)


def test_optimal_policy_is_effect_over_cost():
    spec = _chain()
    obj = ControlObjective(outcome="sales", levers=("price",), cost=2.0)
    assert optimal_policy(spec, obj)["price"] == pytest.approx(0.5)  # 1.0 / 2.0


def test_sign_flip_lever_has_zero_marginal_effect():
    # The thesis seed: a regime sign-flip makes the lever useless to a regime-blind (marginal) plan.
    spec = _sign_flip()
    obj = default_objective(spec)
    assert lever_effects(spec, obj)["price"] == pytest.approx(0.0, abs=1e-9)
    assert optimal_policy(spec, obj)["price"] == pytest.approx(0.0, abs=1e-9)


def test_optimal_policy_has_near_zero_regret_and_beats_doing_nothing():
    spec = _chain()
    obj = default_objective(spec)
    optimal = grade_control(spec, obj, optimal_policy(spec, obj), seed=7)
    zero = grade_control(spec, obj, {"price": 0.0}, seed=7)
    assert optimal.regret == pytest.approx(0.0, abs=0.05)
    assert zero.regret > 0.4  # the optimum is worth ~0.5 reward over doing nothing
    assert zero.achieved_reward < optimal.achieved_reward


def test_grade_controller_runs_a_pluggable_controller():
    spec = _chain()
    obj = default_objective(spec)
    assert isinstance(_FixedController({"price": 1.0}), Controller)
    report = grade_controller(spec, obj, _FixedController({"price": 1.0}), seed=7)
    assert report.regret == pytest.approx(0.0, abs=0.05)  # it played the optimum


def test_regime_aware_optima_differ_by_regime():
    spec = _sign_flip()
    obj = default_objective(spec)
    assert regime_optimal_policy(spec, obj, set())["price"] == pytest.approx(1.0)  # R off: +1
    assert regime_optimal_policy(spec, obj, {"R"})["price"] == pytest.approx(-1.0)  # R on: -1


def test_regime_blind_policy_suffers_under_perturbation():
    # The stay-optimal thesis: the regime-blind optimum (price=0 here) is fine on average but loses
    # ~0.5 reward in EACH regime, while a regime-aware controller would score 0 regret.
    spec = _sign_flip()
    obj = default_objective(spec)
    static = optimal_policy(spec, obj)  # {"price": 0.0}
    report = regret_under_perturbation(spec, obj, static, seed=7)
    assert report.worst_regret > 0.4
    assert set(report.per_regime) == {"baseline", "R"}
    assert all(r > 0.4 for r in report.per_regime.values())


def test_regime_aware_policy_has_no_regret_in_its_regime():
    # Playing the matching per-regime optimum yields ~0 regret under that regime (the upper bound).
    spec = _sign_flip()
    obj = default_objective(spec)
    aware_on = regime_optimal_policy(spec, obj, {"R"})
    report = regret_under_perturbation(spec, obj, aware_on, seed=7)
    assert report.per_regime["R"] == pytest.approx(0.0, abs=0.05)
