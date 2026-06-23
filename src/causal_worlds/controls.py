"""Synthetic-DAG sanity controls — guard against the "Beware the Simulated DAG!" failure mode.

Reisach et al. (NeurIPS 2021) showed many synthetic SCM benchmarks are gameable: additive-noise data
tends to make a variable's marginal **variance grow along causal edges**, so sorting variables by
variance and regressing each on its lower-variance predecessors (**sortnregress**) recovers the
graph with no discovery at all. ``varsortability`` measures how much a world leaks its order this
way (0.5 = no leak; toward 1.0 = the order is readable off the variances).

A follow-up (Reisach et al., NeurIPS 2023, "A Scale-Invariant Sorting Criterion") showed a *second*,
**scale-invariant** leak: a variable's **predictability from the rest** (its R^2) also tends to grow
along causal edges. ``r2sortability`` measures that and ``R2SortnregressDiscoverer`` is its matching
trivial baseline. Crucially, R^2-sortability **cannot be removed by standardization** — a world that
is varsort-clean (~0.5) but R^2-sortable is still gameable, and the honest fix is to standardize
*during* generation (iSCM), not after. Reviewers in 2026 expect **both** reported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_worlds.protocols import Edges, Substrate
    from causal_worlds.sample import FloatArray

_N = 4000  # rows drawn for the control
_COEFF_EPS = 0.3  # |regression coefficient| above this counts as a recovered edge
_TIE = 0.5  # a tie (equal scores along an edge) contributes this to a sortability score
_VAR_EPS = 1e-12  # below this a column is treated as constant (no explainable variance)


def _sortability_along_edges(node_score: FloatArray, edges: Edges, names: tuple[str, ...]) -> float:
    """Fraction of true edges along which a per-node score increases (the Reisach sortability form).

    0.5 means the score carries no causal-order signal; toward 1.0 means the order is readable from
    it alone. Shared by ``varsortability`` (score = variance) and ``r2sortability`` (score = R^2).
    """
    if not edges:
        return _TIE
    index = {name: i for i, name in enumerate(names)}
    score = 0.0
    for src, dst in edges:
        lo, hi = node_score[index[src]], node_score[index[dst]]
        score += 1.0 if lo < hi else _TIE if lo == hi else 0.0
    return score / len(edges)


def varsortability(data: FloatArray, edges: Edges, names: tuple[str, ...]) -> float:
    """Fraction of true directed edges along which marginal variance increases (Reisach et al.).

    0.5 means variance carries no causal-order signal; values toward 1.0 mean the order is readable
    from the variances alone — i.e. the world is trivially gameable by ``sortnregress``. Removable
    post-hoc by standardizing each column's variance.
    """
    return _sortability_along_edges(data.var(axis=0), edges, names)


def _r2_per_variable(data: FloatArray) -> FloatArray:
    """R^2 of regressing each variable on all the others (Reisach et al. 2023; scale-invariant)."""
    centered = data - data.mean(axis=0)
    n_vars = data.shape[1]
    r2 = np.zeros(n_vars, dtype=np.float64)
    for target in range(n_vars):
        ss_total = float((centered[:, target] ** 2).sum())
        if ss_total < _VAR_EPS:
            continue  # constant column -> nothing to explain, leave R^2 at 0
        others = [col for col in range(n_vars) if col != target]
        design = centered[:, others]
        coefficients, *_ = np.linalg.lstsq(design, centered[:, target], rcond=None)
        residual = centered[:, target] - design @ coefficients
        r2[target] = 1.0 - float((residual**2).sum()) / ss_total
    return r2


def r2sortability(data: FloatArray, edges: Edges, names: tuple[str, ...]) -> float:
    """Fraction of true directed edges along which a variable's R^2 (predictability) increases.

    The scale-invariant analogue of varsortability (Reisach et al. 2023): 0.5 means predictability
    carries no causal-order signal; toward 1.0 means it does. Unlike varsortability, this leak
    **survives standardization**, so it must be reported alongside it.
    """
    return _sortability_along_edges(_r2_per_variable(data), edges, names)


def _regress_in_order(
    data: FloatArray, order: list[int], names: tuple[str, ...], coeff_eps: float
) -> Edges:
    """Regress each variable on its predecessors in ``order`` (the shared sortnregress core).

    Shared by both baselines; only the ordering criterion (variance vs R^2) differs.
    """
    centered = data - data.mean(axis=0)
    edges: set[tuple[str, str]] = set()
    for position, target in enumerate(order):
        predecessors = order[:position]
        if not predecessors:
            continue
        design = centered[:, predecessors]
        coefficients, *_ = np.linalg.lstsq(design, centered[:, target], rcond=None)
        edges.update(
            (names[pred], names[target])
            for pred, coeff in zip(predecessors, coefficients, strict=True)
            if abs(float(coeff)) > coeff_eps
        )
    return frozenset(edges)


def _sortnregress(data: FloatArray, names: tuple[str, ...], coeff_eps: float) -> Edges:
    """The trivial baseline: order by increasing variance, regress each var on its predecessors."""
    order = [int(i) for i in np.argsort(data.var(axis=0))]
    return _regress_in_order(data, order, names, coeff_eps)


def _r2_sortnregress(data: FloatArray, names: tuple[str, ...], coeff_eps: float) -> Edges:
    """The scale-invariant baseline: order by increasing R^2, then regress on predecessors."""
    order = [int(i) for i in np.argsort(_r2_per_variable(data))]
    return _regress_in_order(data, order, names, coeff_eps)


class SortnregressDiscoverer:
    """The variance-order baseline (Reisach et al.) — high scores expose a gameable benchmark."""

    def __init__(self, n: int = _N, coeff_eps: float = _COEFF_EPS) -> None:
        """Store the sample size and the coefficient threshold."""
        self._n = n
        self._coeff_eps = coeff_eps

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:
        """Recover edges by sorting variables by variance and regressing (Discoverer Protocol)."""
        sample = substrate.sample(self._n, seed=seed)
        return _sortnregress(sample.data, substrate.variables, self._coeff_eps)


class R2SortnregressDiscoverer:
    """The scale-invariant R^2-order baseline (Reisach et al. 2023) — survives standardization."""

    def __init__(self, n: int = _N, coeff_eps: float = _COEFF_EPS) -> None:
        """Store the sample size and the coefficient threshold."""
        self._n = n
        self._coeff_eps = coeff_eps

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:
        """Recover edges by sorting variables by R^2 and regressing (Discoverer Protocol)."""
        sample = substrate.sample(self._n, seed=seed)
        return _r2_sortnregress(sample.data, substrate.variables, self._coeff_eps)
