"""Tests for ``world_from_edges`` — the flat-graph -> runnable-WorldSpec compiler.

The load-bearing property is a faithful round-trip: a learned model's structure and slopes go in as
flat edges and come back out of the *compiled* world's own answer-key and control effects
(:func:`answer_key` / :func:`lever_effects`) unchanged — so a world compiled from a recovered model
is a faithful stand-in for it.
"""

import pytest

from causal_worlds.control import default_objective, lever_effects
from causal_worlds.from_edges import WeightedEdge, world_from_edges
from causal_worlds.schema import (
    DanglingReferenceError,
    Role,
    Transform,
    Variable,
    answer_key,
)

# lever -> mediator (1.5) -> kpi (2.0), and lever -> kpi (0.5):
#   total effect lever->kpi = direct 0.5 + mediated 1.5 * 2.0 = 3.5
_VARIABLES: tuple[Variable, ...] = (
    Variable("lever", Role.CONTROLLABLE),
    Variable("mediator", Role.OBSERVABLE),
    Variable("kpi", Role.OUTCOME),
)
_EDGES: tuple[WeightedEdge, ...] = (
    WeightedEdge("lever", "mediator", 1.5),
    WeightedEdge("mediator", "kpi", 2.0),
    WeightedEdge("lever", "kpi", 0.5),
)


def test_edges_sharing_a_target_group_into_one_mechanism() -> None:
    spec = world_from_edges(_VARIABLES, _EDGES)
    by_target = {m.target: m for m in spec.mechanisms}
    assert {t.parent for t in by_target["kpi"].terms} == {"lever", "mediator"}
    assert {t.parent for t in by_target["mediator"].terms} == {"lever"}


def test_structure_round_trips_through_answer_key() -> None:
    spec = world_from_edges(_VARIABLES, _EDGES)
    assert answer_key(spec).edges == frozenset(
        {("lever", "mediator"), ("mediator", "kpi"), ("lever", "kpi")},
    )


def test_magnitude_round_trips_through_lever_effects() -> None:
    spec = world_from_edges(_VARIABLES, _EDGES)
    effects = lever_effects(spec, default_objective(spec))
    assert effects["lever"] == pytest.approx(3.5)  # direct 0.5 + mediated 1.5*2.0


def test_noise_scale_is_applied_to_every_synthesised_mechanism() -> None:
    spec = world_from_edges(_VARIABLES, _EDGES, noise_scale=0.7)
    assert all(m.noise_scale == 0.7 for m in spec.mechanisms)


def test_lag_and_transform_are_preserved_on_the_term() -> None:
    spec = world_from_edges(
        _VARIABLES,
        (WeightedEdge("lever", "kpi", 1.0, lag=2, transform=Transform.TANH),),
    )
    term = next(t for m in spec.mechanisms if m.target == "kpi" for t in m.terms)
    assert term.lag == 2
    assert term.transform is Transform.TANH


def test_compiler_validates_and_rejects_a_dangling_edge() -> None:
    with pytest.raises(DanglingReferenceError):
        world_from_edges(_VARIABLES, (WeightedEdge("lever", "ghost", 1.0),))


def test_compiler_rejects_a_parent_side_dangling_edge() -> None:
    # a parent absent from the variable list is the common learned-model case + a distinct code path
    with pytest.raises(DanglingReferenceError):
        world_from_edges(_VARIABLES, (WeightedEdge("ghost", "kpi", 1.0),))


def test_compiler_rejects_duplicate_edges_rather_than_doubling() -> None:
    # two identical edges would silently sum to a doubled coefficient the answer-key shows once
    with pytest.raises(ValueError, match="duplicate edge"):
        world_from_edges(
            _VARIABLES,
            (WeightedEdge("lever", "kpi", 0.5), WeightedEdge("lever", "kpi", 0.5)),
        )


def test_compiler_rejects_nonpositive_noise_scale() -> None:
    with pytest.raises(ValueError, match="noise_scale"):
        world_from_edges(_VARIABLES, _EDGES, noise_scale=0.0)
