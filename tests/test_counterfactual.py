"""Tests for the Rung-3 counterfactual engine (abduction → action → prediction)."""

import numpy as np
import pytest

from causal_worlds import abduct, counterfactual, predict, worlds
from causal_worlds.counterfactual import TemporalCounterfactualError, _draw_noise, _evaluate
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


def _chain() -> WorldSpec:
    """A 2-variable structural chain: x is a root, y = 2*x + noise."""
    return WorldSpec(
        variables=(Variable("x", Role.CONTROLLABLE), Variable("y", Role.OUTCOME)),
        mechanisms=(Mechanism("y", (Term("x", 2.0),)),),
    )


def _full_factual(spec: WorldSpec, *, seed: int) -> dict[str, float]:
    """A full factual assignment (incl. hidden), via the engine's own noise draw + evaluation."""
    return _evaluate(spec, _draw_noise(spec, np.random.default_rng(seed)), {})


def test_consistency_no_intervention_returns_the_factual():
    # Pearl's consistency axiom: a counterfactual with an empty do() is just the factual world.
    result = counterfactual(_chain(), {}, seed=7)
    assert result.factual == result.counterfactual
    assert result.effect == {"x": 0.0, "y": 0.0}


def test_intervention_propagates_exactly_with_noise_held_fixed():
    # y = 2x; forcing x must move y by exactly 2*delta_x — the SAME unit's noise on y is held fixed.
    result = counterfactual(_chain(), {"x": 5.0}, seed=3)
    delta_x = 5.0 - result.factual["x"]
    assert result.counterfactual["y"] - result.factual["y"] == pytest.approx(2.0 * delta_x)
    assert result.counterfactual["x"] == 5.0  # the forced value is set exactly


def test_is_deterministic_in_the_seed():
    a = counterfactual(_chain(), {"x": 1.0}, seed=11)
    b = counterfactual(_chain(), {"x": 1.0}, seed=11)
    assert a.factual == b.factual
    assert a.counterfactual == b.counterfactual


def test_abduct_then_predict_round_trips_the_factual():
    # Abduction recovers the noise; predicting with no intervention reproduces the observed factual.
    spec = worlds.get("coffee")
    full = _full_factual(spec, seed=5)
    observed = {v.name: full[v.name] for v in spec.variables if not v.hidden}
    assert predict(spec, abduct(spec, full), {}) == observed


def test_hidden_confounder_and_non_descendants_held_fixed():
    # do(footfall) holds every other gear fixed (incl. the hidden local_buzz): variables that do NOT
    # descend from footfall are unchanged; a descendant (sales) moves. That is the counterfactual.
    spec = worlds.get("coffee")
    result = counterfactual(spec, {"footfall": 3.0}, seed=1)
    assert result.counterfactual["footfall"] == 3.0
    assert result.counterfactual["weekend"] == result.factual["weekend"]  # not a descendant
    assert result.counterfactual["price"] == result.factual["price"]  # not a descendant
    assert result.counterfactual["sales"] != result.factual["sales"]  # a descendant -> moves


def test_temporal_world_is_rejected():
    with pytest.raises(TemporalCounterfactualError):
        counterfactual(worlds.get("supply"), {"order": 1.0}, seed=0)
