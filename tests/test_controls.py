"""Tests for the synthetic-DAG controls (varsortability + sortnregress)."""

import numpy as np

from causal_worlds import worlds
from causal_worlds.controls import SortnregressDiscoverer, varsortability
from causal_worlds.protocols import Discoverer
from causal_worlds.sample import build_substrate

_NAMES = ("a", "b")


def test_varsortability_reads_the_variance_order():
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])  # var(a) < var(b)
    assert varsortability(data, frozenset({("a", "b")}), _NAMES) == 1.0  # increases along a->b
    assert varsortability(data, frozenset({("b", "a")}), _NAMES) == 0.0  # decreases along b->a
    assert varsortability(data, frozenset(), _NAMES) == 0.5  # no edges -> no signal


def test_varsortability_counts_ties_as_half():
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # equal variance
    assert varsortability(data, frozenset({("a", "b")}), _NAMES) == 0.5


def test_sortnregress_is_a_discoverer_and_returns_edges():
    assert isinstance(SortnregressDiscoverer(), Discoverer)
    substrate = build_substrate(worlds.get("ecommerce"))
    edges = SortnregressDiscoverer(n=2000).recover(substrate, seed=7)
    assert isinstance(edges, frozenset)
    assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)
