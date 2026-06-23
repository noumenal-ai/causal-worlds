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
_SEED_SPACE = 2**32  # draw independent per-environment seeds from [0, this)


def _intervention_environments(
    substrate: Substrate, n: int, rng: np.random.Generator
) -> tuple[list[FloatArray], list[list[int]]]:
    """An observational env plus one single-target interventional env per variable.

    Returns ``(data, targets)`` for GIES, and is the shared source of the *interventional budget*
    the fair crossover gives every method (pooled, via :func:`pooled_interventional_sample`). Each
    environment gets an INDEPENDENT seed: sharing one seed would alias their noise streams.
    """
    names = substrate.variables
    baseline = substrate.sample(n, seed=int(rng.integers(_SEED_SPACE)))
    binary = {i for i, col in enumerate(baseline.data.T) if len(np.unique(col)) <= _MAX_BINARY}
    data: list[FloatArray] = [baseline.data]
    targets: list[list[int]] = [[]]
    for index, name in enumerate(names):
        values = (
            rng.integers(0, 2, n).astype(float)
            if index in binary
            else rng.normal(_DO_LOC, _DO_SCALE, n)
        )
        env = substrate.sample(n, seed=int(rng.integers(_SEED_SPACE)), do={name: values})
        data.append(env.data)
        targets.append([index])
    return data, targets


def pooled_interventional_sample(substrate: Substrate, *, n: int, seed: int) -> FloatArray:
    """Observational + one single-target interventional env per variable, row-stacked.

    This is how an observational method (PC/FCI/GES) is given the **same interventional budget** as
    GIES and the reference grader, so the crossover compares *methods* on equal data access — the
    information-fairness fix. The method sees the pooled distribution but not the targets.
    """
    rng = np.random.default_rng(seed)
    data, _ = _intervention_environments(substrate, n, rng)
    pooled: FloatArray = np.vstack(data)
    return pooled


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
    """Shared base for the ``causal-learn`` methods (PC/GES/FCI).

    ``interventional`` gives the method the same interventional budget as GIES/the reference grader
    (pooled observational + per-variable do() environments) — the information-fair comparison.
    """

    def __init__(self, n: int = _N, alpha: float = _ALPHA, *, interventional: bool = False) -> None:
        """Store the sample size, CI significance, and whether to pool interventional data."""
        self._n = n
        self._alpha = alpha
        self._interventional = interventional

    def recover(self, substrate: Substrate, *, seed: int) -> Edges:  # pragma: no cover - live run
        """Return directed causal claims (Discoverer Protocol)."""
        return self.detail(substrate, seed=seed).edges

    def detail(self, substrate: Substrate, *, seed: int) -> BaselineResult:  # pragma: no cover
        """Run the method on observational (or pooled interventional) data and parse its graph."""
        data = (
            pooled_interventional_sample(substrate, n=self._n, seed=seed)
            if self._interventional
            else substrate.sample(self._n, seed=seed).data
        )
        graph = self._graph(data)
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

        rng = np.random.default_rng(seed)
        data, targets = _intervention_environments(substrate, self._n, rng)
        adjacency, _ = gies.fit_bic(data, targets)
        return parse_adjacency(np.asarray(adjacency), substrate.variables)


BASELINES: dict[str, type] = {
    "pc": PcDiscoverer,
    "ges": GesDiscoverer,
    "fci": FciDiscoverer,
    "gies": GiesDiscoverer,
}
"""Registry of baseline discoverers by name (each constructs with no required args)."""
