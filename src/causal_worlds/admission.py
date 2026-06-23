"""Grader-independent admission — is a world's declared structure *faithful*, by construction?

The old T3 admitted a world iff the **reference interventional-CI grader recovered it**, which made
the benchmark circular: it was the set of worlds that one grader solves, and we then reported that
grader winning. This module replaces that test with a property of the *declared SCM itself*, in
closed form — no discovery method is run, so admission privileges no method.

For a linear-Gaussian SCM the population covariance is ``Cov = M @ Omega @ M.T`` with
``M = (I - B)^-1`` (``B`` = the coefficient matrix, ``Omega`` = the noise variances).
A declared edge is **faithful** iff it induces a detectable partial correlation given the target's
other observed parents — the parameters do not *cancel* and hide a true edge. Partial correlations
are scale-invariant, so this holds for the iSCM-standardized data the substrate emits.
A regime (sign-flip) edge is checked structurally: the regime must genuinely change a coefficient,
or the switch edge is spurious.

A world that is faithful by construction is recoverable in principle by *some* correct method; that
is the right, grader-agnostic bar for admission. How well any particular discoverer does on it is
then reported separately, on equal footing.
"""

from dataclasses import dataclass

import numpy as np

from causal_worlds.schema import Mechanism, WorldSpec, answer_key

type FloatArray = np.ndarray

_ROOT_VAR = 1.0  # exogenous non-regime root ~ N(0, 1)
_REGIME_VAR = 0.25  # a regime switch ~ Bernoulli(0.5) has variance 0.25
_FAITHFUL_TAU = 0.05  # min |partial correlation| for a declared edge to count as detectable
_REGIME_TAU = 0.1  # min coefficient change for a regime edge to genuinely modulate


@dataclass(frozen=True, slots=True)
class FaithfulnessReport:
    """Whether a spec's declared structure is faithful by construction (grader-independent)."""

    faithful: bool
    min_partial_corr: float
    weakest_edge: tuple[str, str] | None
    regime_ok: bool
    reason: str


def _regime_vars(spec: WorldSpec) -> set[str]:
    """Names used as a binary regime switch by some mechanism."""
    return {m.regime for m in spec.mechanisms if m.regime is not None}


def _observed_linear_parents(
    mechanism: Mechanism, observed: set[str], regimes: set[str]
) -> list[str]:
    """Observed lag-0 *linear* parents of a mechanism (not the regime switch, not hidden vars)."""
    parents = {
        term.parent
        for term in (*mechanism.terms, *(mechanism.regime_terms or ()))
        if term.lag == 0 and term.parent in observed and term.parent not in regimes
    }
    return sorted(parents)


def population_covariance(spec: WorldSpec) -> tuple[tuple[str, ...], FloatArray]:
    """Closed-form covariance of the linear-Gaussian SCM over *all* variables (base regime branch).

    Uses the default (``terms``) branch for the linear coefficients; the regime switch enters as an
    exogenous binary, not a linear term (its effect is modulation, checked separately).
    """
    names = tuple(v.name for v in spec.variables)
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    coeff = np.zeros((n, n), dtype=np.float64)
    noise = np.full(n, _ROOT_VAR, dtype=np.float64)
    regimes = _regime_vars(spec)
    for name in regimes:
        if name in index:
            noise[index[name]] = _REGIME_VAR
    for mechanism in spec.mechanisms:
        row = index[mechanism.target]
        noise[row] = mechanism.noise_scale**2
        for term in mechanism.terms:
            if term.lag == 0 and term.parent != mechanism.target:
                coeff[row, index[term.parent]] += term.coeff
    inverse = np.linalg.inv(np.eye(n) - coeff)
    covariance: FloatArray = inverse @ np.diag(noise) @ inverse.T
    return names, covariance


def _partial_correlation(cov: FloatArray, i: int, j: int, given: list[int]) -> float:
    """Partial correlation of variables ``i`` and ``j`` given ``given`` (from a covariance)."""
    block = [i, j, *given]
    precision = np.linalg.pinv(cov[np.ix_(block, block)])
    denominator = np.sqrt(precision[0, 0] * precision[1, 1])
    if denominator < np.finfo(np.float64).eps:
        return 0.0
    return float(-precision[0, 1] / denominator)


def _regime_modulates(mechanism: Mechanism, tau: float) -> bool:
    """True if the regime branch changes at least one coefficient enough to be a real edge."""
    if mechanism.regime_terms is None:
        return True
    default = {term.parent: term.coeff for term in mechanism.terms}
    return any(
        abs(term.coeff - default.get(term.parent, 0.0)) >= tau for term in mechanism.regime_terms
    )


def check_faithfulness(
    spec: WorldSpec, *, tau: float = _FAITHFUL_TAU, regime_tau: float = _REGIME_TAU
) -> FaithfulnessReport:
    """Check that every declared edge is detectable by construction (grader-independent T3).

    A linear edge is faithful iff ``|partial correlation| >= tau`` given the target's other observed
    parents; a regime (switch) edge is faithful iff the regime changes a coeff by ``regime_tau``.
    """
    names, cov = population_covariance(spec)
    index = {name: i for i, name in enumerate(names)}
    observed = {v.name for v in spec.variables if not v.hidden}
    regimes = _regime_vars(spec)

    weakest = 1.0
    weakest_edge: tuple[str, str] | None = None
    for mechanism in spec.mechanisms:
        if mechanism.target not in observed:
            continue
        parents = _observed_linear_parents(mechanism, observed, regimes)
        for parent in parents:
            others = [index[p] for p in parents if p != parent]
            value = abs(_partial_correlation(cov, index[parent], index[mechanism.target], others))
            if value < weakest:
                weakest, weakest_edge = value, (parent, mechanism.target)

    regime_ok = all(_regime_modulates(m, regime_tau) for m in spec.mechanisms)
    edges_faithful = weakest_edge is None or weakest >= tau
    if not edges_faithful:
        reason = f"unfaithful edge {weakest_edge}: |partial corr| {weakest:.3f} < {tau}"
    elif not regime_ok:
        reason = "spurious regime edge: a regime branch changes no coefficient"
    else:
        reason = "faithful"
    return FaithfulnessReport(
        faithful=edges_faithful and regime_ok,
        min_partial_corr=weakest if weakest_edge is not None else 1.0,
        weakest_edge=weakest_edge,
        regime_ok=regime_ok,
        reason=reason,
    )


def is_nontrivial(spec: WorldSpec) -> bool:
    """Structural non-triviality: the graph has edges but is not complete (any-guess would win)."""
    key = answer_key(spec)
    n_observed = sum(1 for v in spec.variables if not v.hidden)
    possible = n_observed * (n_observed - 1)
    if not key.edges or possible == 0:
        return False
    return len(key.edges) / possible < 1.0
