"""The SCM substrate: compile a validated :class:`WorldSpec` into a deterministic executable world.

Sampling is the functional core — pure and seeded, so ``(spec, n, seed, do)`` is reproducible.
Hidden variables drive the data but are not emitted. A variable used as a mechanism ``regime`` is
sampled as a binary {0, 1} switch; other roots are standard normal; the rest follow their mechanism.

Two sampling regimes, chosen by whether any term carries a lag:

* **Cross-sectional** (all lags 0) — rows are i.i.d.; computed vectorized in topological order.
* **Temporal** (some lag >= 1) — rows are timesteps; each is computed from the current step's lag-0
  parents and earlier steps' lagged parents, with a burn-in discarded so the series settles.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from causal_worlds.schema import Mechanism, Term, WorldSpec, validate

type FloatArray = NDArray[np.float64]

_ROOT_SCALE = 1.0  # an exogenous root (no mechanism, not a regime switch) ~ N(0, 1)
_REGIME_ON = 0.5  # a regime switch variable is "on" where its value exceeds this
_BURN_IN = 200  # temporal: discard this many leading steps so the series settles
_STD_EPS = 1e-9  # columns below this std are left as-is (avoid divide-by-zero)
_MAX_BINARY = 2  # columns with <= this many unique values are left as-is (regimes/switches)


def _standardize_column(column: FloatArray) -> FloatArray:
    """Z-score a continuous column; return a binary/regime column ({0, 1}) untouched.

    Standardizing a regime switch would turn its {0, 1} values into z-scores and break the grader's
    regime-stratification; and a binary column carries no order giveaway anyway. So we only rescale
    columns with more than two distinct values (and never a (near-)constant one).
    """
    if len(np.unique(column)) <= _MAX_BINARY:
        return column
    std = float(column.std())
    if std < _STD_EPS:
        return column
    standardized: FloatArray = (column - column.mean()) / std
    return standardized


def _standardize(data: FloatArray) -> FloatArray:
    """Post-hoc: z-score every continuous column of an emitted matrix (used for temporal series)."""
    out: FloatArray = data.astype(np.float64, copy=True)
    for col in range(data.shape[1]):
        out[:, col] = _standardize_column(data[:, col])
    return out


def _as_column(value: float | FloatArray, n: int) -> FloatArray:
    """A do() value as a length-n column: an array is used as-is, a scalar is broadcast."""
    if isinstance(value, np.ndarray):
        arr: FloatArray = value.astype(np.float64)
        return arr
    full: FloatArray = np.full(n, float(value), dtype=np.float64)
    return full


@dataclass(frozen=True, slots=True)
class Sample:
    """One sampled environment: the observed data plus which variables were intervened on."""

    variables: tuple[str, ...]
    data: FloatArray
    intervened: frozenset[str]


def _contemporaneous(mechanism: Mechanism) -> set[str]:
    """Parents read at the current timestep: lag-0 terms plus the regime switch (set the order)."""
    parents = {term.parent for term in mechanism.terms if term.lag == 0}
    parents |= {term.parent for term in (mechanism.regime_terms or ()) if term.lag == 0}
    if mechanism.regime is not None:
        parents.add(mechanism.regime)
    return parents


def _topological_order(spec: WorldSpec) -> list[str]:
    """Variable names ordered contemporaneous-parents-before-children (DFS post-order)."""
    parents_of: dict[str, set[str]] = {variable.name: set() for variable in spec.variables}
    for mechanism in spec.mechanisms:
        parents_of[mechanism.target] |= _contemporaneous(mechanism)

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


def _max_lag(spec: WorldSpec) -> int:
    """The largest lag anywhere in the spec (0 means the world is purely cross-sectional)."""
    lags = [
        term.lag
        for mechanism in spec.mechanisms
        for term in (*mechanism.terms, *(mechanism.regime_terms or ()))
    ]
    return max(lags, default=0)


class ScmSubstrate:
    """A deterministic SCM world compiled from a *validated* :class:`WorldSpec`."""

    def __init__(self, spec: WorldSpec, *, standardize: bool = True) -> None:
        """Compile a validated spec. ``standardize`` standardizes emitted variables (the default).

        Cross-sectional worlds use **internal standardization (iSCM, Ormaniec et al. 2024)**: each
        continuous variable is z-scored *as it is generated*, in topological order, so children read
        unit-variance parents and neither marginal variance nor predictability (R^2) can compound
        along the causal order. This removes *both* the varsortability and the scale-invariant
        R^2-sortability giveaways that mere post-hoc standardization cannot. Temporal worlds, where
        per-step standardization is ill-defined, fall back to post-hoc column z-scoring. Regime
        switches stay {0, 1}; standardization is affine per column, preserving the CI relationships
        the interventional-CI grader relies on.
        """
        self._order = _topological_order(spec)
        self._mechanisms = {mechanism.target: mechanism for mechanism in spec.mechanisms}
        self._regime_vars = {m.regime for m in spec.mechanisms if m.regime is not None}
        self._observed = tuple(v.name for v in spec.variables if not v.hidden)
        self._max_lag = _max_lag(spec)
        self._standardize = standardize

    @property
    def variables(self) -> tuple[str, ...]:
        """The observed variable names, in the column order of sampled data."""
        return self._observed

    def sample(
        self, n: int, *, seed: int, do: Mapping[str, float | FloatArray] | None = None
    ) -> Sample:
        """Sample ``n`` rows; ``do`` fixes variables to constants or per-row arrays."""
        rng = np.random.default_rng(seed)
        forced: dict[str, float | FloatArray] = dict(do) if do else {}
        if self._max_lag == 0:
            data = self._sample_cross_sectional(n, forced, rng)  # iSCM applied in-loop
        else:
            data = self._sample_temporal(n, forced, rng)
            if self._standardize:
                data = _standardize(data)  # post-hoc: per-step iSCM is ill-defined for series
        return Sample(variables=self._observed, data=data, intervened=frozenset(forced))

    # -- cross-sectional (i.i.d. rows) -------------------------------------------------------------

    def _sample_cross_sectional(
        self, n: int, forced: dict[str, float | FloatArray], rng: np.random.Generator
    ) -> FloatArray:
        """Vectorized i.i.d. sampling (every term is lag-0), with iSCM standardization in-loop.

        Standardizing each continuous variable in topological order *before* its children read it is
        what makes the world iSCM: variance and R^2 cannot accumulate along the causal order. Forced
        (``do``) variables are left exactly as the intervention set them.
        """
        values: dict[str, FloatArray] = {}
        for name in self._order:
            if name in forced:
                values[name] = _as_column(forced[name], n)
                continue
            if name in self._mechanisms:
                column = self._apply(self._mechanisms[name], values, n, rng)
            else:
                column = self._sample_root(name, n, rng)
            values[name] = _standardize_column(column) if self._standardize else column
        data: FloatArray = np.column_stack([values[name] for name in self._observed])
        return data

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
        """Sum ``coeff * parent`` over the terms (lag-0 only path)."""
        acc: FloatArray = np.zeros(n, dtype=np.float64)
        for term in terms:
            acc = acc + term.coeff * values[term.parent]
        return acc

    # -- temporal (rows are timesteps) -------------------------------------------------------------

    def _sample_temporal(
        self, n: int, forced: dict[str, float | FloatArray], rng: np.random.Generator
    ) -> FloatArray:
        """Sequential sampling with a burn-in; lagged terms read earlier timesteps from history."""
        total = n + _BURN_IN
        forced_cols = {name: self._forced_column(value, n) for name, value in forced.items()}
        history: dict[str, FloatArray] = {v: np.zeros(total, dtype=np.float64) for v in self._order}
        for t in range(total):
            for name in self._order:
                if name in forced_cols:
                    history[name][t] = forced_cols[name][t]
                elif name in self._mechanisms:
                    history[name][t] = self._step(self._mechanisms[name], history, t, rng)
                else:
                    history[name][t] = self._root_step(name, rng)
        data: FloatArray = np.column_stack([history[name][_BURN_IN:] for name in self._observed])
        return data

    @staticmethod
    def _forced_column(value: float | FloatArray, n: int) -> FloatArray:
        """A do() column over burn-in + n steps; arrays are front-padded with their first value."""
        if isinstance(value, np.ndarray):
            return np.concatenate([np.full(_BURN_IN, value[0], dtype=np.float64), value])
        return np.full(_BURN_IN + n, float(value), dtype=np.float64)

    def _root_step(self, name: str, rng: np.random.Generator) -> float:
        """One timestep of an exogenous root: binary if a regime switch, else standard normal."""
        if name in self._regime_vars:
            return float(rng.integers(0, 2))
        return float(rng.normal(0.0, _ROOT_SCALE))

    def _step(
        self, mechanism: Mechanism, history: dict[str, FloatArray], t: int, rng: np.random.Generator
    ) -> float:
        """One timestep of a mechanism: regime-switched lagged-linear combo plus Gaussian noise."""
        out = self._linear_at(mechanism.terms, history, t)
        regime, regime_terms = mechanism.regime, mechanism.regime_terms
        if regime is not None and regime_terms is not None and history[regime][t] > _REGIME_ON:
            out = self._linear_at(regime_terms, history, t)
        return out + float(rng.normal(0.0, mechanism.noise_scale))

    @staticmethod
    def _linear_at(terms: tuple[Term, ...], history: dict[str, FloatArray], t: int) -> float:
        """Sum ``coeff * parent[t - lag]`` over the terms; reads before t=0 are 0."""
        acc = 0.0
        for term in terms:
            past = t - term.lag
            if past >= 0:
                acc += term.coeff * float(history[term.parent][past])
        return acc


def build_substrate(spec: WorldSpec, *, standardize: bool = True) -> ScmSubstrate:
    """Validate a spec and compile it into an executable substrate (the Factory).

    ``standardize`` (default) standardizes emitted variables — iSCM in-loop for cross-sectional
    worlds, post-hoc for temporal — to remove the varsortability and R^2-sortability giveaways.
    """
    validate(spec)
    return ScmSubstrate(spec, standardize=standardize)
