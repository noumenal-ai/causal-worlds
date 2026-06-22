"""Tests for the pydantic boundary model and JSON serialization."""

from causal_worlds import worlds
from causal_worlds.serde import WorldSpecModel, spec_from_json, spec_to_json


def test_spec_round_trips_through_boundary_model():
    # coffee exercises every field: hidden vars, multi-term mechanisms, a regime + regime_terms.
    spec = worlds.get("coffee")
    assert WorldSpecModel.from_spec(spec).to_spec() == spec


def test_spec_round_trips_through_json():
    for name in worlds.names():
        spec = worlds.get(name)
        assert spec_from_json(spec_to_json(spec)) == spec


def test_no_regime_serializes_with_null_regime_terms():
    # ecommerce has no regime, so regime_terms must round-trip as None, not ().
    spec = worlds.get("ecommerce")
    restored = spec_from_json(spec_to_json(spec))
    assert all(mechanism.regime_terms is None for mechanism in restored.mechanisms)
