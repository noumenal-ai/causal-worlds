"""Tests for the scoring functions."""

from causal_worlds.evaluation import directed_shd, f1, score, skeleton_shd
from causal_worlds.schema import AnswerKey


def test_perfect_recovery():
    truth = frozenset({("a", "b"), ("b", "c")})
    assert directed_shd(truth, truth) == 0
    assert skeleton_shd(truth, truth) == 0
    assert f1(truth, truth) == 1.0


def test_reversed_edge_counts_once():
    truth = frozenset({("a", "b")})
    recovered = frozenset({("b", "a")})
    assert directed_shd(recovered, truth) == 1
    assert skeleton_shd(recovered, truth) == 0  # adjacency is right, only direction wrong


def test_missing_and_extra():
    truth = frozenset({("a", "b"), ("b", "c")})
    recovered = frozenset({("a", "b"), ("a", "c")})  # miss b->c, extra a->c
    assert directed_shd(recovered, truth) == 2


def test_f1_empty_cases():
    empty: frozenset[tuple[str, str]] = frozenset()
    assert f1(empty, empty) == 1.0
    assert f1(empty, frozenset({("a", "b")})) == 0.0


def test_score_flags_confounded_edge():
    key = AnswerKey(
        edges=frozenset({("price", "demand"), ("demand", "sales")}),
        confounded=frozenset({frozenset({"overtime", "sales"})}),
    )
    # a grader that (wrongly) reports the confounded pair as a causal edge
    recovered = frozenset({("price", "demand"), ("demand", "sales"), ("overtime", "sales")})
    report = score(recovered, key)
    assert report.confounded_reported == 1
    assert report.n_truth == 2
    assert report.directed_shd == 1  # the spurious overtime->sales edge


def test_score_clean_recovery():
    key = AnswerKey(
        edges=frozenset({("price", "demand"), ("demand", "sales")}),
        confounded=frozenset({frozenset({"overtime", "sales"})}),
    )
    recovered = frozenset({("price", "demand"), ("demand", "sales")})
    report = score(recovered, key)
    assert report.directed_shd == 0
    assert report.f1 == 1.0
    assert report.confounded_reported == 0
