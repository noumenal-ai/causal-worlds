"""Tests for additive-nonlinear mechanisms (issue #10): ``X = Σ coeff·transform(parent) + noise``.

Covers the new :class:`Transform` (the math), sampling and counterfactuals under a transform (the
two functional cores), the serde/anonymize/viz plumbing, the control guard, and the built-in
``braking`` world — whose whole point is that a ``speed**2`` mechanism is invisible to linear
correlation yet fully causal.
"""

import numpy as np
import pytest

from causal_worlds import (
    NonlinearControlError,
    Transform,
    anonymize_spec,
    build_substrate,
    counterfactual,
    has_nonlinear_terms,
    spec_from_json,
    spec_to_json,
    to_dot,
    to_mermaid,
    worlds,
)
from causal_worlds.control import default_objective, optimal_policy
from causal_worlds.counterfactual import abduct, predict
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec, answer_key


def _xy(transform: Transform, coeff: float, noise: float) -> WorldSpec:
    """``x`` (controllable) drives ``y`` (outcome) via ``coeff·transform(x) + noise``."""
    return WorldSpec(
        variables=(Variable("x", Role.CONTROLLABLE), Variable("y", Role.OUTCOME)),
        mechanisms=(Mechanism("y", (Term("x", coeff, transform=transform),), noise_scale=noise),),
    )


# --- Transform: the math --------------------------------------------------------------------------
def test_transform_apply_scalar() -> None:
    assert Transform.IDENTITY.apply(3.0) == 3.0
    assert Transform.SQUARE.apply(3.0) == 9.0
    assert Transform.CUBE.apply(2.0) == 8.0
    assert Transform.ABS.apply(-2.0) == 2.0
    assert Transform.RELU.apply(-1.5) == 0.0
    assert Transform.RELU.apply(1.5) == 1.5
    assert Transform.TANH.apply(0.0) == 0.0


def test_transform_apply_array() -> None:
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(Transform.SQUARE.apply(x), [4.0, 0.0, 9.0])
    np.testing.assert_allclose(Transform.RELU.apply(x), [0.0, 0.0, 3.0])


def test_has_nonlinear_terms() -> None:
    assert has_nonlinear_terms(worlds.get("braking"))
    assert not has_nonlinear_terms(worlds.get("coffee"))


# --- Sampling under a transform -------------------------------------------------------------------
def test_square_mechanism_is_deterministic_and_nonlinear() -> None:
    spec = _xy(Transform.SQUARE, 2.0, noise=0.0)
    sub = build_substrate(spec, standardize=False)
    sample = sub.sample(2000, seed=0)
    x = sample.data[:, sub.variables.index("x")]
    y = sample.data[:, sub.variables.index("y")]
    np.testing.assert_allclose(y, 2.0 * x**2, atol=1e-9)


def test_square_breaks_linear_correlation_but_not_intervention() -> None:
    spec = worlds.get("braking")
    sub = build_substrate(spec, standardize=False)
    bd = sub.variables.index("braking_distance")
    sp = sub.variables.index("speed")
    sample = sub.sample(60_000, seed=0)
    corr = abs(float(np.corrcoef(sample.data[:, sp], sample.data[:, bd])[0, 1]))
    assert corr < 0.05  # speed² is ~uncorrelated with speed: a linear method sees nothing
    hi = float(sub.sample(60_000, seed=1, do={"speed": 2.0}).data[:, bd].mean())
    lo = float(sub.sample(60_000, seed=1, do={"speed": 0.0}).data[:, bd].mean())
    assert hi - lo > 1.0  # but the intervention reveals a large causal effect


# --- Counterfactuals stay exact (additive noise => closed-form abduction) -------------------------
def test_abduction_then_prediction_is_exact_under_nonlinearity() -> None:
    spec = _xy(Transform.SQUARE, 2.0, noise=0.3)
    factual = {"x": 3.0, "y": 18.0}  # y = 2·3² exactly => recovered noise must be 0
    noise = abduct(spec, factual)
    assert noise["y"] == pytest.approx(0.0)
    predicted = predict(spec, noise, {"x": 4.0})
    assert predicted["y"] == pytest.approx(2.0 * 4.0**2)  # 32, the v²-law counterfactual


def test_counterfactual_on_nonlinear_world_is_deterministic() -> None:
    spec = worlds.get("braking")
    a = counterfactual(spec, do={"speed": 2.0}, seed=3)
    b = counterfactual(spec, do={"speed": 2.0}, seed=3)
    assert a.effect == b.effect
    assert a.effect["braking_distance"] != 0.0


# --- Plumbing: serde, anonymize, viz --------------------------------------------------------------
def test_serde_round_trips_transform() -> None:
    spec = worlds.get("braking")
    restored = spec_from_json(spec_to_json(spec))
    assert has_nonlinear_terms(restored)
    assert restored == spec


def test_anonymize_preserves_transform() -> None:
    spec = worlds.get("braking")
    anon, _mapping = anonymize_spec(spec)
    assert has_nonlinear_terms(anon)  # relabeling names must not silently linearize a world


def test_viz_labels_the_nonlinearity() -> None:
    spec = worlds.get("braking")
    assert "square" in to_mermaid(spec)
    assert "square" in to_dot(spec)


# --- Control guard: the closed-form optimum is linear-only ----------------------------------------
def test_control_rejects_a_nonlinear_world() -> None:
    spec = worlds.get("braking")
    with pytest.raises(NonlinearControlError):
        optimal_policy(spec, default_objective(spec))


# --- The built-in braking world -------------------------------------------------------------------
def test_braking_answer_key_keeps_the_nonlinear_edge_and_confounded_pair() -> None:
    key = answer_key(worlds.get("braking"))
    assert ("speed", "braking_distance") in key.edges
    assert frozenset({"road_grip", "visibility"}) in key.confounded


# --- Temporal path also honors transforms (covers _combine_at / _deterministic_at) ----------------
def test_temporal_nonlinear_sampling_runs() -> None:
    spec = WorldSpec(
        variables=(
            Variable("u", Role.CONTROLLABLE),
            Variable("z", Role.OUTCOME),
        ),
        mechanisms=(Mechanism("z", (Term("u", 0.5, lag=1, transform=Transform.SQUARE),)),),
    )
    sample = build_substrate(spec).sample(50, seed=0)
    assert sample.data.shape == (50, 2)
    assert np.isfinite(sample.data).all()
