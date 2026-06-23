"""Tests for the synthetic-DAG controls (varsortability/sortnregress + R^2 variants)."""

import numpy as np

from causal_worlds import worlds
from causal_worlds.controls import (
    R2SortnregressDiscoverer,
    SortnregressDiscoverer,
    r2sortability,
    varsortability,
)
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


def test_r2sortability_reads_the_predictability_order():
    # b = a + small noise: a is the (less predictable) root, b the (more predictable) child.
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 5000)
    b = a + rng.normal(0.0, 0.1, 5000)
    data = np.column_stack([a, b])
    # R^2(a from b) ≈ R^2(b from a) here, but along the true edge a->b R^2 should not decrease.
    assert r2sortability(data, frozenset({("a", "b")}), _NAMES) >= 0.5
    assert r2sortability(data, frozenset(), _NAMES) == 0.5  # no edges -> no signal


def test_r2sortability_is_scale_invariant():
    # Standardizing each column leaves R^2-sortability unchanged (the whole point of the metric).
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 5000)
    b = 2.0 * a + rng.normal(0.0, 0.5, 5000)
    c = b + rng.normal(0.0, 0.5, 5000)
    data = np.column_stack([a, b, c])
    names = ("a", "b", "c")
    edges = frozenset({("a", "b"), ("b", "c")})
    raw = r2sortability(data, edges, names)
    standardized = (data - data.mean(axis=0)) / data.std(axis=0)
    assert abs(r2sortability(standardized, edges, names) - raw) < 1e-9


def test_r2sortnregress_is_a_discoverer_and_returns_edges():
    assert isinstance(R2SortnregressDiscoverer(), Discoverer)
    substrate = build_substrate(worlds.get("ecommerce"))
    edges = R2SortnregressDiscoverer(n=2000).recover(substrate, seed=7)
    assert isinstance(edges, frozenset)
    assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)
