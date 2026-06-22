"""Standard causal-discovery methods wrapped behind the :class:`Discoverer` Protocol.

The baseline suite for the crossover experiment: do the *standard* methods fail on our worlds
(hidden confounder + regime sign-flip) where the reference interventional-CI grader succeeds? We
wrap PC, GES, FCI (``causal-learn``) and GIES (``gies``) behind our own Protocol + adapter, never
leaking their types. The stack is lazy-imported (the ``discover`` extra), so the package imports and
CI run without it; the parsing logic is pure and unit-tested.

Graph encoding (``causal-learn`` ``GeneralGraph.graph``): ``g[a][b]`` is the endpoint at node ``a``
of edge (a,b) — ``-1`` tail, ``1`` arrowhead, ``2`` circle, ``0`` none. So ``g[i][j]=-1, g[j][i]=1``
is ``i -> j``; both ``1`` is bidirected (confounding); both ``-1`` is undirected. ``gies`` returns a
0/1 adjacency where ``A[i][j]=1`` is ``i -> j`` and ``A[i][j]=A[j][i]=1`` is undirected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_worlds.protocols import Edges, Substrate
    from causal_worlds.sample import FloatArray

_N = 4000  # observational rows drawn for a baseline
_ALPHA = 0.01  # CI test significance for PC/FCI
_DO_LOC = 0.5  # GIES interventions: continuous do-value mean
_DO_SCALE = 1.5  # GIES interventions: continuous do-value spread
_MAX_BINARY = 2  # a column with <= this many unique values is treated as a binary (regime) var
_ARROW = 1  # arrowhead endpoint code (causal-learn)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """A discovered structure, richer than directed edges so the crossover can score fairly.

    ``edges`` = directed causal claims; ``bidirected`` = pairs the method marks as confounded (the
    correct verdict for a hidden-confounded pair, *not* a causal claim); ``skeleton`` = all
    adjacencies (orientation-agnostic) for a fair comparison across CPDAG/PAG outputs.
    """

    edges: Edges
    bidirected: frozenset[frozenset[str]]
    skeleton: frozenset[frozenset[str]]


def parse_endpoint_matrix(graph: FloatArray, names: tuple[str, ...]) -> BaselineResult:
    """Parse a ``causal-learn`` endpoint matrix into a :class:`BaselineResult` (pure)."""
    edges: set[tuple[str, str]] = set()
    bidirected: set[frozenset[str]] = set()
    skeleton: set[frozenset[str]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ep_i, ep_j = int(graph[i][j]), int(graph[j][i])
            if ep_i == 0 and ep_j == 0:
                continue
            a, b = names[i], names[j]
            skeleton.add(frozenset((a, b)))
            if ep_i == _ARROW and ep_j == _ARROW:
                bidirected.add(frozenset((a, b)))
            else:
                _orient(edges, a, b, ep_i, ep_j)
    return BaselineResult(frozenset(edges), frozenset(bidirected), frozenset(skeleton))


def _orient(edges: set[tuple[str, str]], a: str, b: str, ep_a: int, ep_b: int) -> None:
    """Add the oriented edge(s) for one adjacency: toward a lone arrowhead, else both directions."""
    if ep_b == _ARROW and ep_a != _ARROW:
        edges.add((a, b))
    elif ep_a == _ARROW and ep_b != _ARROW:
        edges.add((b, a))
    else:  # tail-tail or circle-circle — an unoriented adjacency
        edges.add((a, b))
        edges.add((b, a))


def parse_adjacency(adjacency: FloatArray, names: tuple[str, ...]) -> BaselineResult:
    """Parse a ``gies`` 0/1 adjacency (no bidirected edges) into a result (pure)."""
    edges: set[tuple[str, str]] = set()
    skeleton: set[frozenset[str]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out_ij, out_ji = int(adjacency[i][j]), int(adjacency[j][i])
            if out_ij == 0 and out_ji == 0:
                continue
            a, b = names[i], names[j]
            skeleton.add(frozenset((a, b)))
            if out_ij == 1 and out_ji == 0:
                edges.add((a, b))
            elif out_ji == 1 and out_ij == 0:
                edges.add((b, a))
            else:
                edges.add((a, b))
                edges.add((b, a))
    return BaselineResult(frozenset(edges), frozenset(), frozenset(skeleton))


class _CausalLearnDiscoverer:
    """Shared base for the ``causal-learn`` observational methods (PC/GES/FCI)."""

    def __init__(self, n: int = _N, alpha: float = _ALPHA) -> None:
        """Store the sample size and CI significance."""
        self._n = n
        self._alpha = alpha

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:  # pragma: no cover - live run
        """Return directed causal claims (Discoverer Protocol)."""
        return self.detail(substrate, seed=seed).edges

    def detail(self, substrate: Substrate, *, seed: int) -> BaselineResult:  # pragma: no cover
        """Run the method on an observational sample and parse its graph."""
        sample = substrate.sample(self._n, seed=seed)
        graph = self._graph(sample.data)
        return parse_endpoint_matrix(graph, substrate.variables)

    def _graph(self, data: FloatArray) -> FloatArray:  # pragma: no cover - third-party call
        raise NotImplementedError


class PcDiscoverer(_CausalLearnDiscoverer):
    """The PC algorithm (constraint-based, observational; assumes causal sufficiency)."""

    def _graph(self, data: FloatArray) -> FloatArray:  # pragma: no cover - third-party call
        from causallearn.search.ConstraintBased.PC import pc  # noqa: PLC0415

        result = pc(data, self._alpha, "fisherz", verbose=False, show_progress=False)
        return np.asarray(result.G.graph)


class GesDiscoverer(_CausalLearnDiscoverer):
    """Greedy Equivalence Search (score-based, observational; assumes causal sufficiency)."""

    def _graph(self, data: FloatArray) -> FloatArray:  # pragma: no cover - third-party call
        from causallearn.search.ScoreBased.GES import ges  # noqa: PLC0415

        return np.asarray(ges(data)["G"].graph)


class FciDiscoverer(_CausalLearnDiscoverer):
    """FCI (constraint-based, latent-aware — can mark a pair bidirected/confounded)."""

    def _graph(self, data: FloatArray) -> FloatArray:  # pragma: no cover - third-party call
        from causallearn.search.ConstraintBased.FCI import fci  # noqa: PLC0415

        graph, _ = fci(
            data,
            independence_test_method="fisherz",
            alpha=self._alpha,
            verbose=False,
            show_progress=False,
        )
        return np.asarray(graph.graph)


class GiesDiscoverer:
    """GIES — score-based and *interventional*, but assumes causal sufficiency.

    The hidden confounder L is outside its model. Fed an observational environment plus one
    single-target intervention environment per variable.
    """

    def __init__(self, n: int = _N) -> None:
        """Store the per-environment sample size."""
        self._n = n

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:  # pragma: no cover - live run
        """Return directed causal claims (Discoverer Protocol)."""
        return self.detail(substrate, seed=seed).edges

    def detail(self, substrate: Substrate, *, seed: int) -> BaselineResult:  # pragma: no cover
        """Build interventional environments, run GIES, and parse its adjacency."""
        import gies  # noqa: PLC0415

        names = substrate.variables
        rng = np.random.default_rng(seed)
        baseline = substrate.sample(self._n, seed=seed)
        binary = {i for i, col in enumerate(baseline.data.T) if len(np.unique(col)) <= _MAX_BINARY}
        data = [baseline.data]
        targets: list[list[int]] = [[]]
        for index, name in enumerate(names):
            values = (
                rng.integers(0, 2, self._n).astype(float)
                if index in binary
                else rng.normal(_DO_LOC, _DO_SCALE, self._n)
            )
            data.append(substrate.sample(self._n, seed=seed, do={name: values}).data)
            targets.append([index])
        adjacency, _ = gies.fit_bic(data, targets)
        return parse_adjacency(np.asarray(adjacency), names)


BASELINES: dict[str, type] = {
    "pc": PcDiscoverer,
    "ges": GesDiscoverer,
    "fci": FciDiscoverer,
    "gies": GiesDiscoverer,
}
"""Registry of baseline discoverers by name (each constructs with no required args)."""
