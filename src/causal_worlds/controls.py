"""Synthetic-DAG sanity controls — guard against the "Beware the Simulated DAG!" failure mode.

Reisach et al. (NeurIPS 2021) showed many synthetic SCM benchmarks are gameable: additive-noise data
tends to make a variable's marginal **variance grow along causal edges**, so sorting variables by
variance and regressing each on its lower-variance predecessors (**sortnregress**) recovers it with
no discovery at all. ``varsortability`` measures how much a world leaks its order this way
(0.5 = no leak; toward 1.0 = the order is readable off the variances); the
``SortnregressDiscoverer`` is that trivial baseline, so a world's gameability is *measured*.

Reviewers discount synthetic-DAG benchmarks that don't report this; standardizing each column's
variance removes the giveaway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_worlds.protocols import Edges, Substrate
    from causal_worlds.sample import FloatArray

_N = 4000  # rows drawn for the control
_COEFF_EPS = 0.3  # |regression coefficient| above this counts as a recovered edge
_TIE = 0.5  # variance tie contributes this to varsortability


def varsortability(data: FloatArray, edges: Edges, names: tuple[str, ...]) -> float:
    """Fraction of true directed edges along which marginal variance increases (Reisach et al.).

    0.5 means variance carries no causal-order signal; values toward 1.0 mean the order is readable
    from the variances alone — i.e. the world is trivially gameable by ``sortnregress``.
    """
    if not edges:
        return _TIE
    variance = data.var(axis=0)
    index = {name: i for i, name in enumerate(names)}
    score = 0.0
    for src, dst in edges:
        var_src, var_dst = variance[index[src]], variance[index[dst]]
        score += 1.0 if var_src < var_dst else _TIE if var_src == var_dst else 0.0
    return score / len(edges)


def _sortnregress(data: FloatArray, names: tuple[str, ...], coeff_eps: float) -> Edges:
    """The trivial baseline: order by increasing variance, regress each var on its predecessors."""
    centered = data - data.mean(axis=0)
    order = list(np.argsort(data.var(axis=0)))  # increasing variance
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
