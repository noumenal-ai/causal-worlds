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
from collections import deque
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


def require_linear_mechanisms(spec: WorldSpec, *, reason: str) -> None:
    """Guard the closed-form (linear-quadratic) control path: raise on a nonlinear world.

    The single authoritative check shared by the optimum solver and the Gymnasium env, so the two
    boundaries can never disagree on what "controllable" means.

    Args:
        spec: The world whose mechanisms must be linear.
        reason: Caller context, woven into the error message.

    Raises:
        NonlinearControlError: ``spec`` has any non-identity transform.
    """
    if has_nonlinear_terms(spec):
        msg = f"this world has nonlinear mechanisms (issue #10): {reason}"
        raise NonlinearControlError(msg)


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
    require_linear_mechanisms(spec, reason="the control optimum is linear-quadratic")
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


# --------------------------------------------------------------------------- #
# Reference baseline controllers (#27) — the control analogue of the sortnregress
# discovery baselines: two calibration points that bracket what the levers can achieve.
# Both estimate each lever's effect from the (raw) substrate, then play u*_i = effect_i / cost.
# --------------------------------------------------------------------------- #
class CorrelationalController:
    """Baseline: set levers from the **observational** regression of the outcome on the levers.

    "Control by correlation" — the control analogue of ``sortnregress``. It runs *no* interventions,
    so on a world with a **hidden** lever↔outcome confounder its effect estimate is biased: **high
    regret here is the signal that the world rewards causal (interventional) understanding** rather
    than mere association. The lower bracket of the achievable range.
    """

    def __init__(self, n: int = _REWARD_N) -> None:
        """``n`` rows of observational data to fit the regression."""
        self._n = n

    def control(
        self, substrate: Substrate, objective: ControlObjective, *, seed: int
    ) -> dict[str, float]:
        """Fit ``outcome ~ levers`` on observational data; play ``u_i = coeff_i / cost``."""
        sample = substrate.sample(self._n, seed=seed)
        index = {name: i for i, name in enumerate(substrate.variables)}
        outcome = sample.data[:, index[objective.outcome]]
        levers = list(objective.levers)
        design = np.column_stack(
            [sample.data[:, index[lever]] for lever in levers] + [np.ones(len(outcome))]
        )
        coeffs, *_ = np.linalg.lstsq(design, outcome, rcond=None)
        return {lever: float(coeffs[i]) / objective.cost for i, lever in enumerate(levers)}


class InterventionalController:
    """Baseline ceiling: per-lever ``do(+1)`` vs ``do(0)`` contrast (the do-calculus reference).

    Estimates each lever's causal effect by *acting* — ``do()`` cuts the lever's incoming edges, so
    the contrast is unbiased by confounding. This is the achievable **ceiling** against which the
    correlational baseline's regret is read; on confounded worlds the two diverge sharply.
    """

    def __init__(self, n: int = _REWARD_N) -> None:
        """``n`` rows per ``do()`` environment."""
        self._n = n

    def control(
        self, substrate: Substrate, objective: ControlObjective, *, seed: int
    ) -> dict[str, float]:
        """Per lever, play ``u_i = (E[outcome|do=1] - E[outcome|do=0]) / cost``."""
        out_i = substrate.variables.index(objective.outcome)
        policy: dict[str, float] = {}
        for lever in objective.levers:
            hi = substrate.sample(self._n, seed=seed, do={lever: 1.0}).data[:, out_i].mean()
            lo = substrate.sample(self._n, seed=seed, do={lever: 0.0}).data[:, out_i].mean()
            policy[lever] = float(hi - lo) / objective.cost
        return policy


# --------------------------------------------------------------------------- #
# Observational identifiability (#27) — does a lever's effect on the outcome admit a
# valid observational backdoor adjustment set? A *hidden* common cause of a lever and
# the outcome makes the effect non-identifiable from any observational method; an
# *observed* confounder is fixable by adjustment; a mediator must not be adjusted for.
# This lets a harness split worlds into "observationally solvable" vs "interventions-required".
# Operates on the contemporaneous (lag-0) causal graph the control optimum lives on.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LeverIdentifiability:
    """Is a lever's outcome-effect recoverable from observational data by backdoor adjustment."""

    lever: str
    identifiable: bool
    adjustment_set: (
        frozenset[str] | None
    )  # a valid observed backdoor set if identifiable, else None


def _lag0_graph(
    spec: WorldSpec,
) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]]]:
    """The contemporaneous directed graph: (nodes, hidden, parents, children) over lag-0 edges."""
    nodes = {v.name for v in spec.variables}
    hidden = {v.name for v in spec.variables if v.hidden}
    parents: dict[str, set[str]] = {n: set() for n in nodes}
    children: dict[str, set[str]] = {n: set() for n in nodes}
    for mechanism in spec.mechanisms:
        ps = {t.parent for t in mechanism.terms if t.lag == 0}
        ps |= {t.parent for t in (mechanism.regime_terms or ()) if t.lag == 0}
        if mechanism.regime is not None:
            ps.add(mechanism.regime)
        for parent in ps:
            parents[mechanism.target].add(parent)
            children[parent].add(mechanism.target)
    return nodes, hidden, parents, children


def _reach(graph: dict[str, set[str]], start: set[str]) -> set[str]:
    """All nodes reachable from ``start`` following ``graph`` edges (excludes the start nodes)."""
    seen: set[str] = set()
    stack = [n for s in start for n in graph[s]]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(graph[node])
    return seen


def _d_connected(
    parents: dict[str, set[str]], children: dict[str, set[str]], x: str, y: str, z: set[str]
) -> bool:
    """Whether ``x`` is d-connected to ``y`` given ``z`` (Koller-Friedman reachability)."""
    anc_z = z | _reach(parents, z)
    queue = deque([(x, "up")])
    visited: set[tuple[str, str]] = set()
    reach: set[str] = set()
    while queue:
        node, direction = queue.popleft()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))
        if node != x:
            reach.add(node)
        if direction == "up" and node not in z:
            queue.extend((p, "up") for p in parents[node])
            queue.extend((c, "down") for c in children[node])
        elif direction == "down":
            if node not in z:
                queue.extend((c, "down") for c in children[node])
            if node in anc_z:  # collider whose descendant (or itself) is in z → path active
                queue.extend((p, "up") for p in parents[node])
    return y in reach


def is_control_identifiable(
    spec: WorldSpec, objective: ControlObjective
) -> dict[str, LeverIdentifiability]:
    """Per lever, whether its outcome-effect is observationally identifiable by backdoor adjustment.

    Uses the complete adjustment criterion: identifiable iff the canonical observed adjustment set
    (observed ancestors of lever and outcome, minus the lever's descendants and the lever/outcome
    themselves) d-separates lever from outcome in the backdoor graph (lever out-edges removed). A
    HIDDEN common cause of lever and outcome leaves an unblockable backdoor path, so it is not
    identifiable; an OBSERVED confounder lands in the set; a MEDIATOR is a lever-descendant and is
    excluded. Lets a harness split worlds into observationally-solvable vs interventions-required.
    (Contemporaneous lag-0 structure.)
    """
    nodes, hidden, parents, children = _lag0_graph(spec)
    observed = nodes - hidden
    out: dict[str, LeverIdentifiability] = {}
    for lever in objective.levers:
        descendants = _reach(children, {lever})
        ancestors = _reach(parents, {lever, objective.outcome})
        candidate = (
            ((ancestors | {lever, objective.outcome}) & observed)
            - descendants
            - {lever, objective.outcome}
        )
        parents_bd = {n: set(ps) for n, ps in parents.items()}
        children_bd = {n: set(cs) for n, cs in children.items()}
        for child in children[lever]:
            parents_bd[child].discard(lever)
        children_bd[lever] = set()
        identifiable = not _d_connected(
            parents_bd, children_bd, lever, objective.outcome, candidate
        )
        out[lever] = LeverIdentifiability(
            lever=lever,
            identifiable=identifiable,
            adjustment_set=frozenset(candidate) if identifiable else None,
        )
    return out
