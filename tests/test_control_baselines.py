"""Tests for the reference baseline controllers (#27) — the control analogue of sortnregress.

`CorrelationalController` (obs regression) and `InterventionalController` (`do()` contrast)
bracket what the levers achieve: on a world with a hidden lever-outcome confounder the correlational
baseline is biased (high regret) while the interventional one stays near-optimal.
"""

from causal_worlds import control_substrate, default_objective, grade_controller, worlds
from causal_worlds.control import CorrelationalController, InterventionalController
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


def _lever_confounded() -> WorldSpec:
    """Hidden ``H`` confounds lever ``L`` and outcome ``Y``; the true do(L) effect is 0.5."""
    return WorldSpec(
        variables=(
            Variable("H", Role.DISTURBANCE, hidden=True),
            Variable("L", Role.CONTROLLABLE),
            Variable("Y", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism("L", (Term("H", 0.9),)),
            Mechanism("Y", (Term("L", 0.5), Term("H", 1.2))),
        ),
    )


def test_controllers_return_a_value_per_lever() -> None:
    spec = worlds.get("ecommerce")
    objective = default_objective(spec)
    for ctrl in (CorrelationalController(), InterventionalController()):
        policy = ctrl.control(control_substrate(spec), objective, seed=0)
        assert set(policy) == set(objective.levers)


def test_both_baselines_are_near_optimal_on_a_clean_world() -> None:
    """ecommerce has no hidden lever↔outcome confounder, so the obs estimate is unbiased."""
    spec = worlds.get("ecommerce")
    objective = default_objective(spec)
    corr = grade_controller(spec, objective, CorrelationalController(), seed=7)
    intv = grade_controller(spec, objective, InterventionalController(), seed=7)
    assert corr.regret < 0.05
    assert intv.regret < 0.05


def test_correlational_baseline_is_fooled_by_a_hidden_lever_confounder() -> None:
    """The bracket diverges: a hidden confounder biases the obs controller but not do()."""
    spec = _lever_confounded()
    objective = default_objective(spec)
    corr = grade_controller(spec, objective, CorrelationalController(), seed=7)
    intv = grade_controller(spec, objective, InterventionalController(), seed=7)
    assert intv.regret < 0.05  # do() recovers the true 0.5 effect → near-optimal
    assert corr.regret > intv.regret + 0.1  # correlation is biased up by H → meaningfully worse
