"""Tests for the baseline graph parsers (pure; no third-party discovery libs needed)."""

import numpy as np

from causal_worlds.baselines import (
    BASELINES,
    FciDiscoverer,
    GiesDiscoverer,
    PcDiscoverer,
    parse_adjacency,
    parse_endpoint_matrix,
)
from causal_worlds.protocols import Discoverer

_NAMES = ("a", "b", "c")


def test_parse_endpoint_matrix_directed_bidirected_and_undirected():
    # a -> b (tail at a, arrow at b); b <-> c (arrows both ends); a -- c (tails both ends).
    g = np.zeros((3, 3))
    g[0][1], g[1][0] = -1, 1  # a -> b
    g[1][2], g[2][1] = 1, 1  # b <-> c
    g[0][2], g[2][0] = -1, -1  # a -- c
    result = parse_endpoint_matrix(g, _NAMES)
    assert ("a", "b") in result.edges
    assert frozenset(("b", "c")) in result.bidirected
    assert frozenset(("b", "c")) not in {frozenset(e) for e in result.edges}  # bidir is not causal
    assert ("a", "c") in result.edges  # unoriented -> both directions
    assert ("c", "a") in result.edges
    assert result.skeleton == {frozenset(("a", "b")), frozenset(("b", "c")), frozenset(("a", "c"))}


def test_parse_adjacency_directed_and_undirected():
    # a -> b ; b -- c ; no edge a-c.
    adj = np.zeros((3, 3))
    adj[0][1] = 1  # a -> b
    adj[1][2], adj[2][1] = 1, 1  # b -- c
    result = parse_adjacency(adj, _NAMES)
    assert ("a", "b") in result.edges
    assert ("b", "c") in result.edges  # undirected -> both directions
    assert ("c", "b") in result.edges
    assert result.bidirected == frozenset()
    assert result.skeleton == {frozenset(("a", "b")), frozenset(("b", "c"))}


def test_empty_graph_parses_to_nothing():
    result = parse_endpoint_matrix(np.zeros((3, 3)), _NAMES)
    assert result.edges == frozenset()
    assert result.skeleton == frozenset()


def test_baselines_satisfy_the_discoverer_protocol():
    assert set(BASELINES) == {"pc", "ges", "fci", "gies"}
    for cls in BASELINES.values():
        assert isinstance(cls(), Discoverer)
    assert isinstance(PcDiscoverer(), Discoverer)
    assert isinstance(FciDiscoverer(), Discoverer)
    assert isinstance(GiesDiscoverer(), Discoverer)
