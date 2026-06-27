"""Stage-2 control — the optimal-policy answer-key, correct-by-construction.

The discovery track grades a method against the *declared graph*. The control track grades a policy
against the *declared optimum*: because the SCM's mechanisms are known, the best the levers can do
is a deterministic function of those mechanisms — an answer-key, exactly like the graph (see
``docs/scope.md`` §1a; this is why the "magnitude problem" dissolves).

We model the simplest faithful control problem — a **linear-quadratic** objective: pick lever values
``u`` to maximise ``E[outcome | do(u)] - (cost/2) * ||u||²``. For a linear-Gaussian SCM a lever's
total effect on the outcome is a path-sum read off ``(I - B)⁻¹`` (levers are intervened, so their
incoming edges are cut); regimes are exogenous Bernoulli switches we marginalise over. The optimum
then decouples per lever: ``u*_i = effect_i / cost``. A controller is graded by **regret** from it.

Control uses the **unstandardized** substrate: iSCM standardization (a discovery-side anti-leakage
device) centres every variable, which would erase the outcome's response to the levers. Magnitudes
are exactly what control needs, so :func:`control_substrate` builds the raw world.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from causal_worlds.errors import CausalWorldsError
from causal_worlds.sample import build_substrate
from causal_worlds.schema import Role, WorldSpec, has_nonlinear_terms

if TYPE_CHECKING:
    from collections.abc import Mapping

    from causal_worlds.protocols import Controller, Substrate

type FloatArray = np.ndarray

_DEFAULT_COST = 1.0  # quadratic lever-cost weight (keeps the objective bounded)
_REWARD_N = 4000  # rows drawn to score a proposed policy's expected reward


class NonlinearControlError(CausalWorldsError):
    """The closed-form control optimum is linear-quadratic and cannot grade a nonlinear world.

    The optimal policy is read off ``(I - B)⁻¹``, which assumes linear mechanisms. A world with any
    non-identity :class:`~causal_worlds.schema.Transform` has no such closed form here; sampling and
    discovery grading still work, but the control answer-key does not yet (issue #10 follow-up).
    """


@dataclass(frozen=True, slots=True)
class ControlObjective:
    """A linear-quadratic control problem over a world: maximise ``E[outcome] - cost/2 * ||u||²``.

    ``levers`` are the controllable variables the policy may set; ``outcome`` is the KPI to raise.
    Both default (via :func:`default_objective`) to the spec's controllable and outcome roles.
    """

    outcome: str
    levers: tuple[str, ...]
    cost: float = _DEFAULT_COST


def default_objective(spec: WorldSpec, *, cost: float = _DEFAULT_COST) -> ControlObjective:
    """Derive the default objective from roles: observed controllables raise the first outcome."""
    levers = tuple(v.name for v in spec.variables if not v.hidden and v.role == Role.CONTROLLABLE)
    outcomes = [v.name for v in spec.variables if not v.hidden and v.role == Role.OUTCOME]
    if not levers or not outcomes:
        msg = "a control objective needs at least one controllable lever and one outcome"
        raise ValueError(msg)
    return ControlObjective(outcome=outcomes[0], levers=levers, cost=cost)


def control_substrate(spec: WorldSpec) -> Substrate:
    """The raw (unstandardized) world a controller acts on — true magnitudes, not z-scores."""
    return build_substrate(spec, standardize=False)


@dataclass(frozen=True, slots=True)
class ControlReport:
    """How a policy did: its reward, the optimum, and the gap (regret >= 0, lower is better)."""

    regret: float
    optimal_reward: float
    achieved_reward: float
    optimal_policy: dict[str, float]


def _regime_vars(spec: WorldSpec) -> tuple[str, ...]:
    """Names used as a binary regime switch (marginalised over when computing total effects)."""
    return tuple({m.regime for m in spec.mechanisms if m.regime is not None})


def _branch_coefficients(
    spec: WorldSpec, regime_on: set[str], index: dict[str, int], levers: set[str]
) -> FloatArray:
    """Coefficient matrix ``B`` for one regime branch, with intervened levers' rows cut to zero."""
    coeff = np.zeros((len(index), len(index)), dtype=np.float64)
    for mechanism in spec.mechanisms:
        if mechanism.target in levers:
            continue  # an intervened lever is exogenous — its mechanism is cut
        on = mechanism.regime is not None and mechanism.regime in regime_on
        terms = (
            mechanism.regime_terms
            if (on and mechanism.regime_terms is not None)
            else mechanism.terms
        )
        for term in terms:
            if term.lag == 0 and term.parent != mechanism.target:
                coeff[index[mechanism.target], index[term.parent]] += term.coeff
    return coeff


def _config_effects(
    spec: WorldSpec, objective: ControlObjective, regime_on: set[str]
) -> dict[str, float]:
    """Total effect of each lever on the outcome for one fixed regime configuration (path-sum)."""
    if has_nonlinear_terms(spec):
        msg = "control optimum is linear-quadratic; this world has nonlinear mechanisms (issue #10)"
        raise NonlinearControlError(msg)
    names = tuple(v.name for v in spec.variables)
    index = {name: i for i, name in enumerate(names)}
    out_i = index[objective.outcome]
    coeff = _branch_coefficients(spec, regime_on, index, set(objective.levers))
    influence = np.linalg.inv(np.eye(len(names)) - coeff)
    return {lever: float(influence[out_i, index[lever]]) for lever in objective.levers}


def regime_configs(spec: WorldSpec) -> list[set[str]]:
    """Every regime switch setting (the ``2^k`` subsets of the regime variables that are 'on')."""
    regimes = _regime_vars(spec)
    return [
        {name for name, flag in zip(regimes, combo, strict=True) if flag}
        for combo in itertools.product((False, True), repeat=len(regimes))
    ]


def lever_effects(spec: WorldSpec, objective: ControlObjective) -> dict[str, float]:
    """Total effect of each lever on the outcome, marginalised over regimes (the answer-key core).

    Each regime configuration's effect is the path-sum ``(I - B)⁻¹[outcome, lever]`` with the levers
    intervened (rows cut); we average over the ``2^k`` equally-likely regime switches.
    """
    per = [_config_effects(spec, objective, on) for on in regime_configs(spec)]
    return {lever: sum(e[lever] for e in per) / len(per) for lever in objective.levers}


def optimal_policy(spec: WorldSpec, objective: ControlObjective) -> dict[str, float]:
    """The (regime-blind) reward-maximising levers — LQ optimum ``u*_i = effect_i / cost``."""
    effects = lever_effects(spec, objective)
    return {lever: effect / objective.cost for lever, effect in effects.items()}


def regime_optimal_policy(
    spec: WorldSpec, objective: ControlObjective, regime_on: set[str]
) -> dict[str, float]:
    """The optimum when the regime is known to be ``regime_on`` (that branch's effects only)."""
    effects = _config_effects(spec, objective, regime_on)
    return {lever: effect / objective.cost for lever, effect in effects.items()}


def expected_reward(
    substrate: Substrate,
    objective: ControlObjective,
    policy: Mapping[str, float],
    *,
    seed: int,
    regime: Mapping[str, float] | None = None,
) -> float:
    """Score a policy on the (raw) world: ``mean(outcome | do(policy)) - cost/2 * ||u||²``.

    ``regime`` pins regime switches to fixed values (a perturbation) via ``do`` — the lever penalty
    excludes them, as they are the environment's state, not the policy's cost.
    """
    do = {lever: float(policy.get(lever, 0.0)) for lever in objective.levers}
    penalty = 0.5 * objective.cost * sum(value**2 for value in do.values())
    if regime:
        do.update({name: float(value) for name, value in regime.items()})
    sample = substrate.sample(_REWARD_N, seed=seed, do=do)
    outcome = sample.data[:, substrate.variables.index(objective.outcome)].mean()
    return float(outcome) - penalty


@dataclass(frozen=True, slots=True)
class PerturbationReport:
    """How a fixed policy holds up as the regime shifts (the stay-optimal-under-perturbation test).

    ``per_regime`` maps each regime setting to the policy's regret there (vs that regime's optimum);
    ``worst_regret`` / ``mean_regret`` summarise across regimes. A regime-blind policy can be best
    on average yet have large worst-case regret when a sign-flipped regime is active.
    """

    worst_regret: float
    mean_regret: float
    per_regime: dict[str, float]


def regret_under_perturbation(
    spec: WorldSpec, objective: ControlObjective, policy: Mapping[str, float], *, seed: int
) -> PerturbationReport:
    """Regret of a fixed ``policy`` against the regime-aware optimum, across every regime shift."""
    substrate = control_substrate(spec)
    regimes = _regime_vars(spec)
    per_regime: dict[str, float] = {}
    for on in regime_configs(spec):
        pin = {name: (1.0 if name in on else 0.0) for name in regimes}
        aware = regime_optimal_policy(spec, objective, on)
        best = expected_reward(substrate, objective, aware, seed=seed, regime=pin)
        got = expected_reward(substrate, objective, policy, seed=seed, regime=pin)
        per_regime["+".join(sorted(on)) or "baseline"] = best - got
    values = list(per_regime.values())
    return PerturbationReport(
        worst_regret=max(values, default=0.0),
        mean_regret=sum(values) / len(values) if values else 0.0,
        per_regime=per_regime,
    )


def grade_control(
    spec: WorldSpec, objective: ControlObjective, policy: Mapping[str, float], *, seed: int
) -> ControlReport:
    """Grade a proposed ``policy`` by regret against the by-construction optimum (raw world)."""
    substrate = control_substrate(spec)
    best = optimal_policy(spec, objective)
    optimal_reward = expected_reward(substrate, objective, best, seed=seed)
    achieved = expected_reward(substrate, objective, policy, seed=seed)
    return ControlReport(
        regret=optimal_reward - achieved,
        optimal_reward=optimal_reward,
        achieved_reward=achieved,
        optimal_policy=best,
    )


def grade_controller(
    spec: WorldSpec, objective: ControlObjective, controller: Controller, *, seed: int
) -> ControlReport:
    """Run a pluggable controller on the raw world and grade its policy by regret."""
    policy = controller.control(control_substrate(spec), objective, seed=seed)
    return grade_control(spec, objective, policy, seed=seed)
