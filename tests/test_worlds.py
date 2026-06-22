"""Tests for the built-in worlds."""

import pytest

from causal_worlds import worlds
from causal_worlds.schema import answer_key, validate


def test_all_builtins_valid_and_have_structure():
    for name in worlds.names():
        spec = worlds.get(name)
        validate(spec)  # must not raise
        assert answer_key(spec).edges  # a non-empty ground-truth graph


def test_coffee_has_a_confounded_pair():
    assert answer_key(worlds.get("coffee")).confounded


def test_unknown_world_raises():
    with pytest.raises(worlds.UnknownWorldError):
        worlds.get("does-not-exist")
