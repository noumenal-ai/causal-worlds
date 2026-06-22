"""Tests for the structural-difficulty metric."""

from causal_worlds import worlds
from causal_worlds.difficulty import structural_difficulty


def test_coffee_has_a_confounder_and_a_sign_flip():
    # hidden L is a common cause of foot/overtime/sales; overtime~sales is the confounded pair;
    # price flips sign on demand across regime R.
    d = structural_difficulty(worlds.get("coffee"))
    assert d.hidden_confounders == 1
    assert d.confounded_pairs == 1
    assert d.sign_flips == 1
    assert d.score == 2.0  # confounded_pairs + sign_flips
    assert 0.0 < d.density < 1.0


def test_ecommerce_is_structurally_easy():
    d = structural_difficulty(worlds.get("ecommerce"))
    assert d.hidden_confounders == 0
    assert d.confounded_pairs == 0
    assert d.sign_flips == 0
    assert d.score == 0.0


def test_coffee_is_harder_than_ecommerce():
    assert (
        structural_difficulty(worlds.get("coffee")).score
        > structural_difficulty(worlds.get("ecommerce")).score
    )
