"""The SCM substrate: compile a validated :class:`WorldSpec` into a deterministic executable world.

Sampling is the functional core — pure and seeded, so ``(spec, n, seed, do)`` is reproducible.
Hidden variables drive the data but are not emitted. A variable used as a mechanism ``regime`` is
sampled as a binary {0, 1} switch; other roots are standard normal; the rest follow their mechanism.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causal_worlds.schema import Mechanism, Term, WorldSpec, validate

type FloatArray = NDArray[np.float64]

_ROOT_SCALE = 1.0  # an exogenous root (no mechanism, not a regime switch) ~ N(0, 1)
_REGIME_ON = 0.5  # a regime switch variable is "on" where its value exceeds this


@dataclass(frozen=True, slots=True)
class Sample:
    """One sampled environment: the observed data plus which variables were intervened on."""

    variables: tuple[str, ...]
    data: FloatArray
    intervened: frozenset[str]


def _mechanism_parents(mechanism: Mechanism) -> set[str]:
    """Every variable that directly drives a mechanism's target (incl. the regime switch)."""
    parents = {term.parent for term in mechanism.terms}
    parents |= {term.parent for term in (mechanism.regime_terms or ())}
    if mechanism.regime is not None:
        parents.add(mechanism.regime)
    return parents


def _topological_order(spec: WorldSpec) -> list[str]:
    """Variable names ordered parents-before-children (DFS post-order; terminates regardless)."""
    parents_of: dict[str, set[str]] = {variable.name: set() for variable in spec.variables}
    for mechanism in spec.mechanisms:
        parents_of[mechanism.target] |= _mechanism_parents(mechanism)

    order: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        """Depth-first post-order visit (parents appended before the node)."""
        if node in visited:
            return
        visited.add(node)
        for parent in parents_of[node]:
            visit(parent)
        order.append(node)

    for variable in spec.variables:
        visit(variable.name)
    return order


class ScmSubstrate:
    """A deterministic SCM world compiled from a *validated* :class:`WorldSpec`."""

    def __init__(self, spec: WorldSpec) -> None:
        """Compile a spec that has already passed :func:`causal_worlds.schema.validate`."""
        self._order = _topological_order(spec)
        self._mechanisms = {mechanism.target: mechanism for mechanism in spec.mechanisms}
        self._regime_vars = {m.regime for m in spec.mechanisms if m.regime is not None}
        self._observed = tuple(v.name for v in spec.variables if not v.hidden)

    @property
    def variables(self) -> tuple[str, ...]:
        """The observed variable names, in the column order of sampled data."""
        return self._observed

    def sample(self, n: int, *, seed: int, do: Mapping[str, float] | None = None) -> Sample:
        """Sample ``n`` rows; ``do`` fixes the named variables (an intervention)."""
        rng = np.random.default_rng(seed)
        forced = dict(do) if do else {}
        values: dict[str, FloatArray] = {}
        for name in self._order:
            if name in forced:
                values[name] = np.full(n, float(forced[name]), dtype=np.float64)
            elif name in self._mechanisms:
                values[name] = self._apply(self._mechanisms[name], values, n, rng)
            else:
                values[name] = self._sample_root(name, n, rng)
        data: FloatArray = np.column_stack([values[name] for name in self._observed])
        return Sample(variables=self._observed, data=data, intervened=frozenset(forced))

    def _sample_root(self, name: str, n: int, rng: np.random.Generator) -> FloatArray:
        """A variable with no mechanism: a binary switch if it's a regime, else standard normal."""
        if name in self._regime_vars:
            out: FloatArray = rng.integers(0, 2, n).astype(np.float64)
            return out
        noise: FloatArray = rng.normal(0.0, _ROOT_SCALE, n)
        return noise

    def _apply(
        self,
        mechanism: Mechanism,
        values: dict[str, FloatArray],
        n: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Evaluate a mechanism: (regime-switched) linear combo of parents plus Gaussian noise."""
        out: FloatArray = self._linear(mechanism.terms, values, n)
        if mechanism.regime is not None and mechanism.regime_terms is not None:
            switched = self._linear(mechanism.regime_terms, values, n)
            on = values[mechanism.regime] > _REGIME_ON
            out = np.where(on, switched, out)
        out = out + rng.normal(0.0, mechanism.noise_scale, n)
        return out

    @staticmethod
    def _linear(terms: tuple[Term, ...], values: dict[str, FloatArray], n: int) -> FloatArray:
        """Sum ``coeff * parent`` over the terms."""
        acc: FloatArray = np.zeros(n, dtype=np.float64)
        for term in terms:
            acc = acc + term.coeff * values[term.parent]
        return acc


def build_substrate(spec: WorldSpec) -> ScmSubstrate:
    """Validate a spec and compile it into an executable substrate (the Factory)."""
    validate(spec)
    return ScmSubstrate(spec)
