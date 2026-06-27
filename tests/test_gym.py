"""Tests for the Gymnasium control-env adapter (the gym extra)."""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from causal_worlds.control import NonlinearControlError, regime_optimal_policy  # noqa: E402
from causal_worlds.gym import ControlEnv  # noqa: E402
from causal_worlds.schema import Mechanism, Role, Term, Transform, Variable, WorldSpec  # noqa: E402


def _sign_flip() -> WorldSpec:
    """price -> sales, +1 in one regime and -1 in the other (regret-under-perturbation world)."""
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


def test_control_env_rejects_a_nonlinear_world_at_construction():
    """The regret signal needs the linear-quadratic optimum, so a nonlinear world must fail fast
    at construction — not crash mid-episode when step() reaches for the optimum (issue #10)."""
    nonlinear = WorldSpec(
        variables=(Variable("price", Role.CONTROLLABLE), Variable("sales", Role.OUTCOME)),
        mechanisms=(Mechanism("sales", (Term("price", 1.0, transform=Transform.SQUARE),)),),
    )
    with pytest.raises(NonlinearControlError):
        ControlEnv(nonlinear)


def test_env_is_a_gymnasium_env_with_the_right_spaces():
    env = ControlEnv(_sign_flip(), horizon=3)
    assert isinstance(env, gymnasium.Env)
    assert env.action_space.shape == (1,)  # one lever (price)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert "regime" in info


def test_episode_terminates_after_horizon_and_reports_regret():
    env = ControlEnv(_sign_flip(), horizon=3)
    env.reset(seed=0)
    steps, terminated = 0, False
    while not terminated:
        _obs, reward, terminated, _truncated, info = env.step(np.zeros(1))
        assert isinstance(reward, float)
        assert info["regret"] >= -0.05  # regret is non-negative up to sampling noise
        steps += 1
    assert steps == 3


def test_regime_aware_play_has_near_zero_regret_but_doing_nothing_does_not():
    # Playing each regime's optimum yields ~0 cumulative regret; the zero action loses every step.
    env = ControlEnv(_sign_flip(), horizon=4)
    env.reset(seed=1)
    aware_regret, terminated = 0.0, False
    while not terminated:
        best = regime_optimal_policy(env._spec, env._objective, set(env._regime_on))  # noqa: SLF001
        action = np.array([best[lever] for lever in env._objective.levers])  # noqa: SLF001
        _obs, _reward, terminated, _trunc, info = env.step(action)
        aware_regret += info["regret"]
    assert aware_regret == pytest.approx(0.0, abs=0.2)

    env.reset(seed=1)
    zero_regret, terminated = 0.0, False
    while not terminated:
        _obs, _reward, terminated, _trunc, info = env.step(np.zeros(1))
        zero_regret += info["regret"]
    assert zero_regret > 1.0  # ~0.5 lost each of 4 steps on the sign-flip world
