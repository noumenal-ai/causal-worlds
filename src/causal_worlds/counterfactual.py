"""Rung 3 of Pearl's ladder — counterfactuals on a declared SCM (abduction → action → prediction).

A counterfactual asks: *we observed this specific case; what **would** the outcome have been had we
acted differently, on that same case?* Pearl's recipe, exact for our declared structural model:

1. **Abduction** — recover the exogenous noise that produced the factual case (``factual[X] -
   mechanism_X(parents)`` per variable; a root's value *is* its noise).
2. **Action** — apply the intervention by graph surgery (the same ``do`` semantics the sampler uses:
   the intervened variable's equation is replaced by the forced value).
3. **Prediction** — re-run the structural equations with the **same** recovered noise but the new
   equation, so the only thing that changes is the consequence of the intervention.

This is a *structural* query, so it works on the raw structural equations (the natural scale of the
mechanisms), not the iSCM-standardized data the benchmark emits. Holding the noise fixed is
what makes it a counterfactual about *the same unit*, not a fresh draw — and **autonomy /
modularity** (each mechanism is independent) is what licenses re-running every other equation
unchanged. Cross-sectional worlds only for now; temporal (trajectory) counterfactuals are future.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from causal_worlds.errors import CausalWorldsError
from causal_worlds.schema import Mechanism, WorldSpec, validate

_REGIME_ON = 0.5  # a regime switch is "on" above this (matches the sampler)
_ROOT_SCALE = 1.0  # an exogenous (non-regime) root ~ N(0, 1)

type Assignment = dict[str, float]


class TemporalCounterfactualError(CausalWorldsError):
    """Counterfactuals on temporal (lagged) worlds are not supported yet."""


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    """A factual case and what *would* have happened under an intervention, on that same case."""

    factual: Assignment
    counterfactual: Assignment
    intervention: dict[str, float]

    @property
    def effect(self) -> Assignment:
        """The per-variable change the intervention caused: ``counterfactual - factual``."""
        return {name: self.counterfactual[name] - self.factual[name] for name in self.factual}


def _mechanisms(spec: WorldSpec) -> dict[str, Mechanism]:
    return {mechanism.target: mechanism for mechanism in spec.mechanisms}


def _is_temporal(spec: WorldSpec) -> bool:
    return any(
        term.lag > 0
        for mechanism in spec.mechanisms
        for term in (*mechanism.terms, *(mechanism.regime_terms or ()))
    )


def _parents(mechanism: Mechanism) -> set[str]:
    """The variables a mechanism reads: its term parents plus any regime switch (lag-0)."""
    parents = {term.parent for term in (*mechanism.terms, *(mechanism.regime_terms or ()))}
    if mechanism.regime is not None:
        parents.add(mechanism.regime)
    return parents


def _topological_order(spec: WorldSpec) -> list[str]:
    """Variable names with parents before children (DFS post-order over contemporaneous parents)."""
    mechanisms = _mechanisms(spec)
    order: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        if node in mechanisms:
            for parent in _parents(mechanisms[node]):
                visit(parent)
        order.append(node)

    for variable in spec.variables:
        visit(variable.name)
    return order


def _deterministic(mechanism: Mechanism, values: Mapping[str, float]) -> float:
    """The noise-free mechanism output: its linear combo, switched to ``regime_terms`` if on."""
    terms = mechanism.terms
    if (
        mechanism.regime is not None
        and mechanism.regime_terms is not None
        and values[mechanism.regime] > _REGIME_ON
    ):
        terms = mechanism.regime_terms
    return sum(term.coeff * values[term.parent] for term in terms)


def _draw_noise(spec: WorldSpec, rng: np.random.Generator) -> Assignment:
    """One unit's exogenous draw: mechanism noise ~ N(0, scale); regime root {0,1}; else N(0,1)."""
    mechanisms = _mechanisms(spec)
    regime_vars = {m.regime for m in spec.mechanisms if m.regime is not None}
    noise: Assignment = {}
    for variable in spec.variables:
        name = variable.name
        if name in mechanisms:
            noise[name] = float(rng.normal(0.0, mechanisms[name].noise_scale))
        elif name in regime_vars:
            noise[name] = float(rng.integers(0, 2))
        else:
            noise[name] = float(rng.normal(0.0, _ROOT_SCALE))
    return noise


def _evaluate(spec: WorldSpec, noise: Mapping[str, float], do: Mapping[str, float]) -> Assignment:
    """Propagate a fixed noise vector through the equations under ``do`` (graph surgery)."""
    mechanisms = _mechanisms(spec)
    values: Assignment = {}
    for name in _topological_order(spec):
        if name in do:  # surgery: the forced value replaces the equation (incoming edges cut)
            values[name] = float(do[name])
        elif name in mechanisms:
            values[name] = _deterministic(mechanisms[name], values) + noise[name]
        else:
            values[name] = noise[name]
    return values


def _observed(spec: WorldSpec, values: Assignment) -> Assignment:
    return {v.name: values[v.name] for v in spec.variables if not v.hidden}


def abduct(spec: WorldSpec, factual: Mapping[str, float]) -> Assignment:
    """Step 1 — recover the exogenous noise that produced a fully-observed ``factual`` unit.

    ``factual`` must give a value for **every** variable, including hidden ones (you have these by
    construction when you generate the world). For a mechanism variable the noise is
    ``factual[X] - mechanism_X(parents)``; for a root it is the value itself.
    """
    mechanisms = _mechanisms(spec)
    return {
        name: (factual[name] - _deterministic(mechanisms[name], factual))
        if name in mechanisms
        else factual[name]
        for name in (v.name for v in spec.variables)
    }


def predict(spec: WorldSpec, noise: Mapping[str, float], do: Mapping[str, float]) -> Assignment:
    """Steps 2+3 — re-run the SCM with a fixed ``noise`` under ``do``; returns observed vars."""
    return _observed(spec, _evaluate(spec, noise, do))


def counterfactual(spec: WorldSpec, do: Mapping[str, float], *, seed: int) -> CounterfactualResult:
    """The ground-truth counterfactual on a sampled unit: factual vs. what-would-have-been.

    Samples one unit's exogenous noise (``seed``), computes the **factual** world (no intervention)
    and the **counterfactual** world under ``do`` from the *same* noise, returning both over the
    observed variables. Because the noise is held fixed, the difference is due solely to the
    intervention — that is the counterfactual.

    Raises:
        TemporalCounterfactualError: if the world is temporal (lagged) — not supported yet.
    """
    validate(spec)
    if _is_temporal(spec):
        msg = "counterfactuals on temporal (lagged) worlds are not supported yet"
        raise TemporalCounterfactualError(msg)
    noise = _draw_noise(spec, np.random.default_rng(seed))
    return CounterfactualResult(
        factual=predict(spec, noise, {}),
        counterfactual=predict(spec, noise, do),
        intervention=dict(do),
    )
