"""Time-series causal-discovery baselines for temporal worlds, behind the TemporalDiscoverer seam.

The temporal analog of :mod:`causal_worlds.baselines`: do the standard TS methods recover the lagged
structure, and do the latent-naive ones (PCMCI+, VARLiNGAM, Granger) fall for a hidden confounder
where a latent-aware one (LPCMCI) does not? Each wraps a third-party method (``tigramite``,
``lingam``, ``statsmodels``) behind our own Protocol + adapter, lazy-imported (the ``temporal``
extra); the graph-parsing logic is pure and unit-tested.

PCMCI graph encoding (``tigramite``): ``graph[i][j][tau]`` is the link from ``i`` at lag ``tau`` to
``j`` at lag 0; ``'-->'`` is a directed edge ``i ->(tau) j``, ``'<->'`` is latent confounding (not a
causal edge). VARLiNGAM: ``adjacency_matrices_[k][j][i]`` is the effect of ``i`` (lag ``k``) on j.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_worlds.protocols import Substrate, TemporalEdges
    from causal_worlds.sample import FloatArray

_N = 4000  # timesteps sampled
_MAX_LAG = 2  # tau_max searched
_ALPHA = 0.05  # CI / Granger significance
_COEFF_EPS = 0.1  # VARLiNGAM: |coefficient| above this counts as an edge
_DIRECTED = "-->"  # tigramite directed-link marker


def parse_pcmci_graph(graph: object, names: tuple[str, ...]) -> TemporalEdges:
    """Parse a tigramite link-matrix of strings into directed lagged edges (pure)."""
    arr = np.asarray(graph, dtype=str)
    n_vars, _, n_tau = arr.shape
    return frozenset(
        (names[i], names[j], tau)
        for i in range(n_vars)
        for j in range(n_vars)
        for tau in range(n_tau)
        if arr[i][j][tau] == _DIRECTED
    )


def parse_varlingam(
    matrices: object, names: tuple[str, ...], eps: float = _COEFF_EPS
) -> TemporalEdges:
    """Parse VARLiNGAM lagged adjacency matrices into directed lagged edges (pure)."""
    mats = np.asarray(matrices)
    n_lags, n_vars, _ = mats.shape
    return frozenset(
        (names[i], names[j], lag)
        for lag in range(n_lags)
        for j in range(n_vars)
        for i in range(n_vars)
        if abs(float(mats[lag][j][i])) > eps and not (lag == 0 and i == j)
    )


class PcmciPlusDiscoverer:
    """PCMCI+ (tigramite) — constraint-based TS discovery; assumes no latent confounders."""

    def __init__(self, n: int = _N, max_lag: int = _MAX_LAG, alpha: float = _ALPHA) -> None:
        """Store sample size, max lag, and CI significance."""
        self._n, self._max_lag, self._alpha = n, max_lag, alpha

    def recover_temporal(self, substrate: Substrate, *, seed: int) -> TemporalEdges:
        """Recover lagged edges (Discoverer Protocol)."""
        return parse_pcmci_graph(
            self._graph(substrate.sample(self._n, seed=seed).data), substrate.variables
        )

    def _graph(self, data: FloatArray) -> object:  # pragma: no cover - third-party call
        from tigramite import data_processing as pp  # noqa: PLC0415
        from tigramite.independence_tests.parcorr import ParCorr  # noqa: PLC0415
        from tigramite.pcmci import PCMCI  # noqa: PLC0415

        names = [str(k) for k in range(data.shape[1])]
        pcmci = PCMCI(dataframe=pp.DataFrame(data, var_names=names), cond_ind_test=ParCorr())
        return pcmci.run_pcmciplus(tau_min=0, tau_max=self._max_lag, pc_alpha=self._alpha)["graph"]


class LpcmciDiscoverer:
    """LPCMCI (tigramite) — latent-aware TS discovery; marks confounding ``<->``, not causal."""

    def __init__(self, n: int = _N, max_lag: int = _MAX_LAG, alpha: float = _ALPHA) -> None:
        """Store sample size, max lag, and CI significance."""
        self._n, self._max_lag, self._alpha = n, max_lag, alpha

    def recover_temporal(self, substrate: Substrate, *, seed: int) -> TemporalEdges:
        """Recover lagged edges (Discoverer Protocol)."""
        return parse_pcmci_graph(
            self._graph(substrate.sample(self._n, seed=seed).data), substrate.variables
        )

    def _graph(self, data: FloatArray) -> object:  # pragma: no cover - third-party call
        from tigramite import data_processing as pp  # noqa: PLC0415
        from tigramite.independence_tests.parcorr import ParCorr  # noqa: PLC0415
        from tigramite.lpcmci import LPCMCI  # noqa: PLC0415

        names = [str(k) for k in range(data.shape[1])]
        lpcmci = LPCMCI(dataframe=pp.DataFrame(data, var_names=names), cond_ind_test=ParCorr())
        return lpcmci.run_lpcmci(tau_max=self._max_lag, pc_alpha=self._alpha)["graph"]


class VarLingamDiscoverer:
    """VARLiNGAM (lingam) — non-Gaussian SVAR discovery; assumes no latent confounders."""

    def __init__(self, n: int = _N, max_lag: int = _MAX_LAG) -> None:
        """Store sample size and the VAR order."""
        self._n, self._max_lag = n, max_lag

    def recover_temporal(self, substrate: Substrate, *, seed: int) -> TemporalEdges:
        """Recover lagged edges (Discoverer Protocol)."""
        return parse_varlingam(
            self._matrices(substrate.sample(self._n, seed=seed).data), substrate.variables
        )

    def _matrices(self, data: FloatArray) -> object:  # pragma: no cover - third-party call
        from lingam import VARLiNGAM  # noqa: PLC0415

        model = VARLiNGAM(lags=self._max_lag)
        model.fit(data)
        return model.adjacency_matrices_


class GrangerDiscoverer:
    """Pairwise Granger causality (statsmodels) — lagged-only; assumes no latent confounders."""

    def __init__(self, n: int = _N, max_lag: int = _MAX_LAG, alpha: float = _ALPHA) -> None:
        """Store sample size, max lag, and significance."""
        self._n, self._max_lag, self._alpha = n, max_lag, alpha

    def recover_temporal(
        self, substrate: Substrate, *, seed: int
    ) -> TemporalEdges:  # pragma: no cover
        """Recover lagged edges by testing each ordered pair for Granger causality."""
        from statsmodels.tsa.stattools import grangercausalitytests  # noqa: PLC0415

        data = substrate.sample(self._n, seed=seed).data
        names = substrate.variables
        edges: set[tuple[str, str, int]] = set()
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                pair = np.column_stack([data[:, j], data[:, i]])  # does i Granger-cause j?
                result = grangercausalitytests(pair, maxlag=self._max_lag)
                for lag in range(1, self._max_lag + 1):
                    if result[lag][0]["ssr_ftest"][1] < self._alpha:
                        edges.add((names[i], names[j], lag))
        return frozenset(edges)


TEMPORAL_BASELINES: dict[str, type] = {
    "pcmci+": PcmciPlusDiscoverer,
    "lpcmci": LpcmciDiscoverer,
    "varlingam": VarLingamDiscoverer,
    "granger": GrangerDiscoverer,
}
"""Registry of time-series baseline discoverers by name."""
