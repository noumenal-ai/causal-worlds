"""Tests for the world-spec IR and the static validation gate."""

import pytest

from causal_worlds.schema import (
    CyclicGraphError,
    DanglingEdgeError,
    Edge,
    Role,
    RoleError,
    Variable,
    WorldSpec,
    validate,
)


def _spec(variables, edges):
    return WorldSpec(variables=tuple(variables), edges=tuple(edges))


def _good_spec():
    return _spec(
        [
            Variable("price", Role.CONTROLLABLE),
            Variable("demand", Role.OBSERVABLE),
            Variable("sales", Role.OUTCOME),
        ],
        [Edge("price", "demand"), Edge("demand", "sales")],
    )


def test_valid_spec_passes():
    validate(_good_spec())  # should not raise


def test_dangling_edge_rejected():
    spec = _spec(
        [Variable("price", Role.CONTROLLABLE), Variable("sales", Role.OUTCOME)],
        [Edge("price", "ghost")],
    )
    with pytest.raises(DanglingEdgeError):
        validate(spec)


def test_cycle_rejected():
    spec = _spec(
        [
            Variable("a", Role.CONTROLLABLE),
            Variable("b", Role.OBSERVABLE),
            Variable("c", Role.OUTCOME),
        ],
        [Edge("a", "b"), Edge("b", "c"), Edge("c", "a")],
    )
    with pytest.raises(CyclicGraphError):
        validate(spec)


def test_missing_controllable_rejected():
    spec = _spec(
        [Variable("demand", Role.OBSERVABLE), Variable("sales", Role.OUTCOME)],
        [Edge("demand", "sales")],
    )
    with pytest.raises(RoleError):
        validate(spec)


def test_missing_outcome_rejected():
    spec = _spec(
        [Variable("price", Role.CONTROLLABLE), Variable("demand", Role.OBSERVABLE)],
        [Edge("price", "demand")],
    )
    with pytest.raises(RoleError):
        validate(spec)


def test_self_loop_is_a_cycle():
    spec = _spec(
        [Variable("price", Role.CONTROLLABLE), Variable("sales", Role.OUTCOME)],
        [Edge("price", "price"), Edge("price", "sales")],
    )
    with pytest.raises(CyclicGraphError):
        validate(spec)
