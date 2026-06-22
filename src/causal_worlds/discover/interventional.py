"""The reference causal-discovery grader: an interventional-CI discoverer.

Validated in the project's spikes: standard observational/score-based methods (PC, GES, GIES, FCI)
fail the hidden-confounder trap, but this interventional rule recovers the directed graph and drops
spurious edges between confounded pairs. It is **spec-blind** — it sees only the ``Substrate``'s
variables and may draw observational + interventional samples.

The rule, uniform over every observed variable ``v``:
  1. **Reachability** — ``do(v)`` randomized; ``w`` is an effect of ``v`` iff ``do(v)`` moves ``w``
     (regime-stratified, so a sign-flip by regime is not cancelled by pooling).
  2. Direct edge v->w iff w still depends on v in the do(v) data given w's discovered ancestors
     (they block indirect paths in and are never w's descendants, so no collider is opened).
Regime (stratifier) columns are detected from data as low-cardinality (binary) columns.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from causal_worlds.protocols import Edges, Substrate
from causal_worlds.sample import FloatArray

_EDGE_R = 0.08  # min |partial corr| / |slope| to call a dependence real
_P_MAX = 1e-3  # max p-value for a dependence
_DO_LOC = 0.5  # randomized continuous intervention ~ N(_DO_LOC, _DO_SCALE)
_DO_SCALE = 1.5
_MIN_STRATUM = 50  # a regime stratum needs at least this many rows to be used
_MAX_BINARY = 2  # a column with <= this many distinct values is treated as a regime/binary switch
_PERFECT_CORR = 1.0
_STD_EPS = 1e-9
_SEED_SPACE = 2**32  # range for drawing independent per-call substrate seeds


def _draw_seed(rng: np.random.Generator) -> int:
    """A fresh substrate seed, independent of the interventions drawn from the same generator."""
    return int(rng.integers(0, _SEED_SPACE))


def _residualize(target: FloatArray, conditioning: FloatArray) -> FloatArray:
    """Residual of ``target`` after regressing out the conditioning columns (with an intercept)."""
    n = len(target)
    design: FloatArray = (
        np.column_stack([np.ones(n), conditioning]) if conditioning.size else np.ones((n, 1))
    )
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual: FloatArray = target - design @ coef
    return residual


def _partial_corr(matrix: FloatArray, i: int, j: int, cond: list[int]) -> tuple[float, float]:
    """Partial correlation of columns ``i`` and ``j`` given ``cond``; returns ``(r, p_value)``."""
    conditioning = matrix[:, cond] if cond else np.empty((len(matrix), 0))
    res_i, res_j = (
        _residualize(matrix[:, i], conditioning),
        _residualize(matrix[:, j], conditioning),
    )
    if res_i.std() < _STD_EPS or res_j.std() < _STD_EPS:
        return 0.0, 1.0
    r = float(np.corrcoef(res_i, res_j)[0, 1])
    if abs(r) >= _PERFECT_CORR:
        return r, 0.0
    dof = max(len(matrix) - len(cond) - 3, 1)
    z_stat = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    return r, math.erfc(abs(z_stat) / math.sqrt(2))


def _slope(x: FloatArray, y: FloatArray) -> float:
    """OLS slope of ``y`` on ``x`` (with an intercept)."""
    design: FloatArray = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1])


def _detect_binary(matrix: FloatArray) -> list[int]:
    """Column indices that look like regime/binary switches (low cardinality)."""
    return [c for c in range(matrix.shape[1]) if len(np.unique(matrix[:, c])) <= _MAX_BINARY]


def _strata(matrix: FloatArray, binary_cols: list[int]) -> Iterator[FloatArray]:
    """Yield the pooled matrix and each binary column's level-slices (if large enough)."""
    yield matrix
    for col in binary_cols:
        for level in (0.0, 1.0):
            stratum = matrix[matrix[:, col] == level]
            if len(stratum) >= _MIN_STRATUM:
                yield stratum


def _strat_effect(matrix: FloatArray, v: int, w: int, binary_cols: list[int]) -> float:
    """Strongest OLS slope of ``w`` on ``v`` across the pooled and per-regime strata."""
    best = 0.0
    for stratum in _strata(matrix, binary_cols):
        slope = _slope(stratum[:, v], stratum[:, w])
        if abs(slope) > abs(best):
            best = slope
    return best


def _strat_dep(
    matrix: FloatArray, v: int, w: int, cond: list[int], binary_cols: list[int]
) -> float:
    """Strongest significant partial correlation of ``v``,``w`` | ``cond`` across strata."""
    best = 0.0
    for stratum in _strata(matrix, binary_cols):
        within = [
            c for c in cond if c not in binary_cols
        ]  # a regime is constant within its stratum
        r, p = _partial_corr(stratum, v, w, within)
        if abs(r) >= _EDGE_R and p <= _P_MAX and abs(r) > abs(best):
            best = r
    return best


def _intervention_values(n: int, rng: np.random.Generator, *, binary: bool) -> FloatArray:
    """Randomized do() values: a coin flip for a regime/binary var, else a wide normal."""
    if binary:
        flips: FloatArray = rng.integers(0, 2, n).astype(np.float64)
        return flips
    spread: FloatArray = rng.normal(_DO_LOC, _DO_SCALE, n)
    return spread


@dataclass(frozen=True, slots=True)
class InterventionalCiDiscoverer:
    """Recover a directed graph by interventional conditional-independence testing."""

    n: int = 8000

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:
        """Recover directed edges over the substrate's observed variables."""
        names = substrate.variables
        m = len(names)
        rng = np.random.default_rng(seed)
        # Each substrate sample gets an independent seed so the world's noise is NOT aliased to the
        # interventions drawn from `rng` (sharing a seed manufactures spurious correlations).
        binary_cols = _detect_binary(substrate.sample(self.n, seed=_draw_seed(rng)).data)

        do_data: dict[int, FloatArray] = {}
        for v in range(m):
            values = _intervention_values(self.n, rng, binary=v in binary_cols)
            do_data[v] = substrate.sample(self.n, seed=_draw_seed(rng), do={names[v]: values}).data

        descendants = _reachability(do_data, m, binary_cols)
        return _direct_edges(names, do_data, descendants, binary_cols)


def _reachability(
    do_data: dict[int, FloatArray], m: int, binary_cols: list[int]
) -> dict[int, set[int]]:
    """Stage 1: ``w`` is a descendant of ``v`` iff ``do(v)`` moves ``w``."""
    descendants: dict[int, set[int]] = {v: set() for v in range(m)}
    for v in range(m):
        matrix = do_data[v]
        for w in range(m):
            if w != v and abs(_strat_effect(matrix, v, w, binary_cols)) >= _EDGE_R:
                descendants[v].add(w)
    return descendants


def _direct_edges(
    names: tuple[str, ...],
    do_data: dict[int, FloatArray],
    descendants: dict[int, set[int]],
    binary_cols: list[int],
) -> Edges:
    """Stage 2: keep ``v -> w`` iff ``w`` depends on ``v`` given ``w``'s discovered ancestors."""
    m = len(names)
    edges: set[tuple[str, str]] = set()
    for v in range(m):
        matrix = do_data[v]
        for w in descendants[v]:
            ancestors = [u for u in range(m) if u not in (v, w) and w in descendants[u]]
            if abs(_strat_dep(matrix, v, w, ancestors, binary_cols)) >= _EDGE_R:
                edges.add((names[v], names[w]))
    return frozenset(edges)
