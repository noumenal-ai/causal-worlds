"""Tests for name anonymization (the Caliper-style anti-cliché control)."""

from causal_worlds import worlds
from causal_worlds.anonymize import anonymize_spec
from causal_worlds.difficulty import structural_difficulty
from causal_worlds.sample import build_substrate
from causal_worlds.schema import answer_key, validate


def test_anonymize_renames_every_variable_to_xn():
    spec = worlds.get("coffee")
    anon, mapping = anonymize_spec(spec)
    assert {v.name for v in anon.variables} == {f"X{i + 1}" for i in range(len(spec.variables))}
    assert set(mapping) == {v.name for v in spec.variables}
    assert "price" not in {v.name for v in anon.variables}  # semantic names are gone


def test_anonymize_preserves_structure_and_validity():
    spec = worlds.get("coffee")
    anon, mapping = anonymize_spec(spec)
    validate(anon)  # still a well-formed spec
    # The graph is isomorphic under the mapping: edges map one-to-one.
    original = answer_key(spec).edges
    mapped = frozenset((mapping[s], mapping[d]) for s, d in original)
    assert answer_key(anon).edges == mapped
    # Structural difficulty is name-blind, so it is identical.
    assert structural_difficulty(anon).score == structural_difficulty(spec).score


def test_anonymized_world_is_still_executable():
    anon, _ = anonymize_spec(worlds.get("coffee"))
    sample = build_substrate(anon).sample(200, seed=1)
    assert sample.data.shape[0] == 200
    assert set(sample.variables) <= {v.name for v in anon.variables}
