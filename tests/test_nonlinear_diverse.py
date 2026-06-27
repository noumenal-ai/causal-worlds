"""Diverse-world validation for additive-nonlinear mechanisms (issue #10).

Beyond the focused unit tests in ``test_nonlinear.py``, this exercises *every* transform and several
distinct world shapes — non-monotone transforms, regime-switched nonlinearity, multi-transform
mechanisms, temporal autoregression, a richly confounded world end-to-end — plus the invariants that
must hold for the feature to be trustworthy: the do() fingerprint each transform should leave, exact
counterfactuals (Pearl's consistency axiom), determinism, and byte-for-byte backward compatibility.
"""

import numpy as np
import pytest

from causal_worlds import (
    anonymize_spec,
    build_substrate,
    grade_spec,
    spec_from_json,
    spec_to_json,
    to_dot,
    to_mermaid,
    worlds,
)
from causal_worlds.counterfactual import abduct, counterfactual, predict
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.schema import (
    Mechanism,
    NonStationaryError,
    Role,
    Term,
    Transform,
    Variable,
    WorldSpec,
    answer_key,
    validate,
)

NONLINEAR = [t for t in Transform if t is not Transform.IDENTITY]


def _xy(transform: Transform, coeff: float = 1.0, noise: float = 0.0) -> WorldSpec:
    """``x`` (controllable) -> ``y`` (outcome) via ``coeff·transform(x) + noise``."""
    return WorldSpec(
        variables=(Variable("x", Role.CONTROLLABLE), Variable("y", Role.OUTCOME)),
        mechanisms=(Mechanism("y", (Term("x", coeff, transform=transform),), noise_scale=noise),),
    )


# --- The do() fingerprint each transform leaves ---------------------------------------------------
# These signatures are *why* a transform is what it is: a discoverer that can intervene sees them.
@pytest.mark.parametrize(
    ("transform", "lo", "mid", "hi"),
    [
        (Transform.IDENTITY, -2.0, 0.0, 2.0),  # linear, odd
        (Transform.SQUARE, 4.0, 0.0, 4.0),  # even: do(-2) == do(+2)
        (Transform.CUBE, -8.0, 0.0, 8.0),  # odd, steep
        (Transform.TANH, -0.96, 0.0, 0.96),  # saturating, bounded
        (Transform.RELU, 0.0, 0.0, 2.0),  # rectified: negatives clipped
        (Transform.ABS, 2.0, 0.0, 2.0),  # even, V-shaped
    ],
)
def test_do_fingerprint_per_transform(
    transform: Transform, lo: float, mid: float, hi: float
) -> None:
    sub = build_substrate(_xy(transform), standardize=False)
    yi = sub.variables.index("y")
    got = {v: float(sub.sample(40_000, seed=1, do={"x": v}).data[:, yi].mean()) for v in (-2, 0, 2)}
    assert got[-2] == pytest.approx(lo, abs=0.05)
    assert got[0] == pytest.approx(mid, abs=0.05)
    assert got[2] == pytest.approx(hi, abs=0.05)


# --- Counterfactuals stay exact for every transform -----------------------------------------------
@pytest.mark.parametrize("transform", NONLINEAR)
def test_counterfactual_roundtrip_exact_per_transform(transform: Transform) -> None:
    spec = _xy(transform, coeff=2.0, noise=0.3)
    factual_x = 1.3
    factual_y = 2.0 * float(transform.apply(factual_x))  # noise-free, so abduct must recover 0
    noise = abduct(spec, {"x": factual_x, "y": factual_y})
    assert noise["y"] == pytest.approx(0.0, abs=1e-9)
    predicted = predict(spec, noise, {"x": 1.5})
    assert predicted["y"] == pytest.approx(2.0 * float(transform.apply(1.5)))


def test_pearl_marginalization_holds_on_nonlinear_world() -> None:
    """Cross-validate the two independent code paths on a nonlinear world: the per-unit
    counterfactual engine (scalar ``_deterministic``) averaged over units must equal the
    interventional ``do()`` effect from the vectorized sampler (``_combine``). Pearl's
    marginalization law — if they disagree, one of the cores is wrong."""
    spec = worlds.get("braking")
    sub = build_substrate(spec, standardize=False)
    bd = sub.variables.index("braking_distance")
    do_effect = float(
        sub.sample(200_000, seed=10, do={"speed": 1.5}).data[:, bd].mean()
        - sub.sample(200_000, seed=10, do={"speed": 0.5}).data[:, bd].mean()
    )
    cf_effect = float(
        np.mean(
            [
                counterfactual(spec, do={"speed": 1.5}, seed=s).counterfactual["braking_distance"]
                - counterfactual(spec, do={"speed": 0.5}, seed=s).counterfactual["braking_distance"]
                for s in range(2000)
            ]
        )
    )
    assert cf_effect == pytest.approx(do_effect, abs=0.1)


@pytest.mark.parametrize("transform", NONLINEAR)
def test_consistency_axiom_per_transform(transform: Transform) -> None:
    """Pearl's consistency: intervening a unit to the value it already had changes nothing."""
    spec = _xy(transform, coeff=1.7, noise=0.5)
    factual = {"x": 0.8, "y": 1.7 * float(transform.apply(0.8)) + 0.4}  # arbitrary realized noise
    noise = abduct(spec, factual)
    rerun = predict(spec, noise, {"x": factual["x"]})  # do(x = its own factual value)
    assert rerun["y"] == pytest.approx(factual["y"])


# --- Multi-transform and regime-switched nonlinearity ---------------------------------------------
def test_mixed_transforms_in_one_mechanism_are_exact() -> None:
    spec = WorldSpec(
        variables=(
            Variable("a", Role.CONTROLLABLE),
            Variable("b", Role.CONTROLLABLE),
            Variable("c", Role.DISTURBANCE),
            Variable("y", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism(
                "y",
                (
                    Term("a", 0.5),
                    Term("b", 0.9, transform=Transform.SQUARE),
                    Term("c", 0.4, transform=Transform.TANH),
                ),
                noise_scale=0.0,
            ),
        ),
    )
    sub = build_substrate(spec, standardize=False)
    sample = sub.sample(3000, seed=7)
    col = {v: sample.data[:, sub.variables.index(v)] for v in sub.variables}
    expected = 0.5 * col["a"] + 0.9 * col["b"] ** 2 + 0.4 * np.tanh(col["c"])
    np.testing.assert_allclose(col["y"], expected, atol=1e-9)


def test_regime_switches_the_functional_form() -> None:
    """A regime can flip a term from linear to nonlinear, not just flip its sign."""
    spec = WorldSpec(
        variables=(
            Variable("R", Role.DISTURBANCE),
            Variable("x", Role.CONTROLLABLE),
            Variable("y", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism(
                "y",
                (Term("x", 1.0),),  # linear off-regime
                regime="R",
                regime_terms=(Term("x", 1.0, transform=Transform.SQUARE),),  # square on-regime
                noise_scale=0.0,
            ),
        ),
    )
    sub = build_substrate(spec, standardize=False)
    data = sub.sample(40_000, seed=2)
    col = {v: data.data[:, sub.variables.index(v)] for v in sub.variables}
    off = col["R"] < 0.5
    on = ~off
    assert abs(np.corrcoef(col["x"][off], col["y"][off])[0, 1]) == pytest.approx(1.0, abs=0.02)
    assert abs(np.corrcoef(col["x"][on], col["y"][on])[0, 1]) < 0.1  # linear corr vanishes
    assert np.corrcoef(col["x"][on] ** 2, col["y"][on])[0, 1] == pytest.approx(1.0, abs=0.02)


# --- Temporal nonlinearity: stationarity is enforced at validate() time ---------------------------
def test_cross_variable_lagged_nonlinearity_is_finite() -> None:
    """A *cross-variable* lagged nonlinear edge (no self-feedback) is fine — it can't diverge."""
    spec = WorldSpec(
        variables=(Variable("u", Role.CONTROLLABLE), Variable("z", Role.OUTCOME)),
        mechanisms=(Mechanism("z", (Term("u", 0.5, lag=1, transform=Transform.SQUARE),)),),
    )
    data = build_substrate(spec).sample(100, seed=0).data
    assert np.isfinite(data).all()


def test_bounded_nonlinear_self_loop_is_allowed_and_finite() -> None:
    """A *bounded* transform (tanh) in a self-loop keeps the feedback bounded, so it is admitted
    at any coefficient and stays finite."""
    spec = WorldSpec(
        variables=(Variable("u", Role.CONTROLLABLE), Variable("z", Role.OUTCOME)),
        mechanisms=(Mechanism("z", (Term("z", 3.0, lag=1, transform=Transform.TANH),)),),
    )
    validate(spec)  # bounded => stationary, no rejection
    assert np.isfinite(build_substrate(spec).sample(100, seed=0).data).all()


def test_unbounded_nonlinear_self_loop_is_rejected_at_validate() -> None:
    """An explosive nonlinear self-loop is rejected at authoring time, not silently inf later."""
    spec = WorldSpec(
        variables=(Variable("u", Role.CONTROLLABLE), Variable("z", Role.OUTCOME)),
        mechanisms=(Mechanism("z", (Term("z", 1.5, lag=1, transform=Transform.SQUARE),)),),
    )
    with pytest.raises(NonStationaryError):
        validate(spec)


def test_unbounded_nonlinear_multivariable_cycle_is_rejected() -> None:
    """The check generalizes past self-loops: a multi-variable temporal cycle (a→b→a) with an
    unbounded transform on it has no stationarity guarantee and is rejected at validate."""
    spec = WorldSpec(
        variables=(Variable("a", Role.CONTROLLABLE), Variable("b", Role.OUTCOME)),
        mechanisms=(
            Mechanism(
                "a", (Term("b", 0.9, lag=1, transform=Transform.SQUARE),)
            ),  # a_t = .9 b²_{t-1}
            Mechanism("b", (Term("a", 0.5, lag=1),)),  # b_t = .5 a_{t-1}  → closes the cycle
        ),
    )
    with pytest.raises(NonStationaryError):
        validate(spec)


def test_acyclic_unbounded_nonlinearity_is_allowed() -> None:
    """An unbounded transform NOT on a cycle (a feed-forward chain) is fine and stays finite."""
    spec = WorldSpec(
        variables=(
            Variable("u", Role.CONTROLLABLE),
            Variable("v", Role.OBSERVABLE),
            Variable("w", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism("v", (Term("u", 0.8, lag=1, transform=Transform.SQUARE),)),
            Mechanism("w", (Term("v", 0.5, lag=1, transform=Transform.CUBE),)),
        ),
    )
    validate(spec)  # u→v→w is a DAG, no feedback, so unbounded transforms are safe
    assert np.isfinite(build_substrate(spec).sample(100, seed=0).data).all()


def test_explosive_linear_autoregression_is_rejected_at_validate() -> None:
    """The linear half of the caveat: total AR load >= 1 is rejected too."""
    spec = WorldSpec(
        variables=(Variable("u", Role.CONTROLLABLE), Variable("z", Role.OUTCOME)),
        mechanisms=(Mechanism("z", (Term("z", 1.2, lag=1),)),),
    )
    with pytest.raises(NonStationaryError):
        validate(spec)


def test_stationary_linear_autoregression_is_allowed() -> None:
    """The supply built-in and any sum|coeff| < 1 self-loop must still validate."""
    spec = WorldSpec(
        variables=(Variable("u", Role.CONTROLLABLE), Variable("z", Role.OUTCOME)),
        mechanisms=(Mechanism("z", (Term("z", 0.4, lag=1), Term("z", 0.5, lag=2))),),
    )
    validate(spec)  # 0.4 + 0.5 = 0.9 < 1, stationary
    validate(worlds.get("supply"))


# --- Determinism and backward compatibility -------------------------------------------------------
def test_nonlinear_sampling_is_deterministic() -> None:
    spec = _xy(Transform.CUBE, coeff=0.6, noise=0.3)
    a = build_substrate(spec).sample(1000, seed=3).data
    b = build_substrate(spec).sample(1000, seed=3).data
    np.testing.assert_array_equal(a, b)


def test_identity_transform_is_byte_identical_to_unspecified() -> None:
    explicit = build_substrate(_xy(Transform.IDENTITY, 0.7, 0.3)).sample(500, seed=5).data
    default = (
        build_substrate(
            WorldSpec(
                variables=(Variable("x", Role.CONTROLLABLE), Variable("y", Role.OUTCOME)),
                mechanisms=(Mechanism("y", (Term("x", 0.7),), noise_scale=0.3),),
            )
        )
        .sample(500, seed=5)
        .data
    )
    np.testing.assert_array_equal(explicit, default)


# --- Plumbing across every transform --------------------------------------------------------------
@pytest.mark.parametrize("transform", NONLINEAR)
def test_serde_and_viz_per_transform(transform: Transform) -> None:
    spec = _xy(transform, coeff=0.8, noise=0.2)
    assert spec_from_json(spec_to_json(spec)) == spec  # round-trip preserves the transform
    assert transform.value in to_mermaid(spec)
    assert transform.value in to_dot(spec)


# --- A richly confounded, regime-switched, multi-nonlinear world, end to end ----------------------
def _diverse_world() -> WorldSpec:
    """Hidden ``h`` confounds two nonlinear children; a regime reshapes the outcome's lever term."""
    return WorldSpec(
        variables=(
            Variable("h", Role.DISTURBANCE, hidden=True),
            Variable("season", Role.DISTURBANCE),
            Variable("lever", Role.CONTROLLABLE),
            Variable("sat", Role.OBSERVABLE),  # saturating in h
            Variable("rect", Role.OBSERVABLE),  # rectified in h (confounded sibling of sat)
            Variable("load", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism("sat", (Term("h", 1.0, transform=Transform.TANH),)),
            Mechanism("rect", (Term("h", 1.0, transform=Transform.RELU),)),
            Mechanism(
                "load",
                (Term("lever", 0.6, transform=Transform.SQUARE), Term("sat", 0.5)),
                regime="season",
                regime_terms=(Term("lever", -0.6), Term("sat", 0.5)),  # off-season flips lever
            ),
        ),
    )


def test_diverse_confounded_nonlinear_world_end_to_end() -> None:
    spec = _diverse_world()
    validate(spec)  # must not raise

    key = answer_key(spec)
    assert ("lever", "load") in key.edges
    assert ("sat", "load") in key.edges
    assert frozenset({"sat", "rect"}) in key.confounded  # share hidden h, no edge between them

    data = build_substrate(spec).sample(4000, seed=0).data
    assert np.isfinite(data).all()

    assert spec_from_json(spec_to_json(spec)) == spec
    anon, _ = anonymize_spec(spec)
    assert spec_from_json(spec_to_json(anon)) == anon  # nonlinear anon world still round-trips

    report = grade_spec(spec, InterventionalCiDiscoverer())
    assert report.n_truth == len(key.edges)  # the answer-key the grader scores against is intact
