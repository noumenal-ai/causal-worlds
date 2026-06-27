"""Gymnasium adapter — drive the Stage-2 control benchmark as an RL environment.

A thin :class:`gymnasium.Env` over the control loop: each step the **regime can shift** (a
perturbation), the agent sets **lever values** (the action), and is rewarded by the control
objective under the *current* regime. Reward and the per-step optimum reuse the same
correct-by-construction machinery as :func:`causal_worlds.control.grade_control`, so cumulative
regret against ``info["optimal_reward"]`` is the stay-optimal-under-perturbation score.

The agent sees, each step, the observed-variable means under the current regime (a snapshot it can
use to *detect* a regime shift and adapt) — never the spec, which is the answer. Gymnasium is an
optional dep (``gym`` extra); this module imports it, so ``import causal_worlds`` stays keyless —
import ``causal_worlds.gym`` explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np

from causal_worlds.control import (
    ControlObjective,
    NonlinearControlError,
    control_substrate,
    default_objective,
    expected_reward,
    regime_configs,
    regime_optimal_policy,
)
from causal_worlds.schema import has_nonlinear_terms

if TYPE_CHECKING:
    from causal_worlds.schema import WorldSpec

type FloatArray = np.ndarray

_HORIZON = 10  # steps per episode (the regime may shift between them)
_OBS_N = 1000  # rows summarised into the per-step observation
_ACTION_BOUND = 5.0  # lever values are clipped to [-bound, bound]
_SEED_SPACE = 2**32


def _regime_vars(spec: WorldSpec) -> tuple[str, ...]:
    """Names used as a binary regime switch (the perturbation axis)."""
    return tuple({m.regime for m in spec.mechanisms if m.regime is not None})


class ControlEnv(gymnasium.Env[FloatArray, FloatArray]):
    """A control world as a Gymnasium env: act = lever values, reward = objective under the regime.

    Observation: the observed-variable means under the current (possibly shifted) regime. Action: a
    vector of lever values. ``info`` carries the regime and the regime-aware ``optimal_reward`` so a
    harness can score regret; the agent itself sees only ``obs`` and ``reward``.
    """

    def __init__(
        self,
        spec: WorldSpec,
        objective: ControlObjective | None = None,
        *,
        horizon: int = _HORIZON,
    ) -> None:
        """Compile the (raw) control world and define the action/observation spaces.

        Raises:
            NonlinearControlError: the world has nonlinear mechanisms — the per-step regret signal
                relies on the closed-form (linear-quadratic) optimum, so a nonlinear world is not a
                valid control env yet (issue #10). Rejected here rather than crashing mid-episode.
        """
        if has_nonlinear_terms(spec):
            msg = (
                "ControlEnv needs the linear-quadratic optimum for its regret signal; this world "
                "has nonlinear mechanisms (issue #10). Use it for discovery/counterfactuals."
            )
            raise NonlinearControlError(msg)
        self._spec = spec
        self._objective = objective if objective is not None else default_objective(spec)
        self._substrate = control_substrate(spec)
        self._regimes = _regime_vars(spec)
        self._configs = regime_configs(spec)
        self._horizon = horizon
        self._observed = self._substrate.variables
        self._rng = np.random.default_rng()
        self._regime_on: set[str] = set()
        self._step = 0

        n_levers = len(self._objective.levers)
        n_observed = len(self._observed)
        self.action_space = gymnasium.spaces.Box(
            low=-_ACTION_BOUND, high=_ACTION_BOUND, shape=(n_levers,), dtype=np.float64
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_observed,), dtype=np.float64
        )

    def _draw_seed(self) -> int:
        """An independent sampling seed drawn from the env RNG (advances each step)."""
        return int(self._rng.integers(_SEED_SPACE))

    def _pin(self) -> dict[str, float]:
        """The ``do`` assignment that fixes every regime switch to the current configuration."""
        return {name: (1.0 if name in self._regime_on else 0.0) for name in self._regimes}

    def _observe(self) -> FloatArray:
        """Observed-variable means under the current regime — the agent's snapshot of the world."""
        sample = self._substrate.sample(_OBS_N, seed=self._draw_seed(), do=self._pin())
        means: FloatArray = sample.data.mean(axis=0)
        return means

    def _shift_regime(self) -> None:
        """Draw the next regime configuration (the perturbation)."""
        self._regime_on = self._configs[int(self._rng.integers(len(self._configs)))]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002 - part of the gymnasium reset contract
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Seed the env, pick an initial regime, and return the first observation."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._shift_regime()
        return self._observe(), {"regime": sorted(self._regime_on)}

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Apply the lever values under the current regime; reward = objective; then maybe shift."""
        policy = {
            lever: float(value)
            for lever, value in zip(self._objective.levers, action, strict=False)
        }
        seed = self._draw_seed()
        pin = self._pin()
        reward = expected_reward(self._substrate, self._objective, policy, seed=seed, regime=pin)
        best_policy = regime_optimal_policy(self._spec, self._objective, self._regime_on)
        optimal = expected_reward(
            self._substrate, self._objective, best_policy, seed=seed, regime=pin
        )
        info = {
            "regime": sorted(self._regime_on),
            "optimal_reward": optimal,
            "regret": optimal - reward,
        }
        self._step += 1
        terminated = self._step >= self._horizon
        if not terminated:
            self._shift_regime()
        return self._observe(), reward, terminated, False, info
