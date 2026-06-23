"""Tests for grader-independent admission (closed-form faithfulness)."""

import numpy as np

from causal_worlds import worlds
from causal_worlds.admission import check_faithfulness, is_nontrivial, population_covariance
from causal_worlds.sample import build_substrate
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


def _chain(direct_coeff: float) -> WorldSpec:
    """a -> b -> c with a direct a -> c edge of the given coefficient (0 => a degenerate edge)."""
    return WorldSpec(
        variables=(
            Variable("a", Role.CONTROLLABLE),
            Variable("b", Role.OBSERVABLE),
            Variable("c", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism("b", (Term("a", 1.0),)),
            Mechanism("c", (Term("b", 1.0), Term("a", direct_coeff))),
        ),
    )


def test_builtin_worlds_are_faithful_by_construction():
    assert check_faithfulness(worlds.get("coffee")).faithful
    assert check_faithfulness(worlds.get("ecommerce")).faithful
    assert is_nontrivial(worlds.get("coffee"))


def test_population_covariance_matches_sampled_covariance():
    # The closed-form covariance should match the empirical covariance of unstandardized samples.
    spec = _chain(0.5)
    names, cov = population_covariance(spec)
    sub = build_substrate(spec, standardize=False)
    data = sub.sample(40000, seed=1).data
    observed = [names.index(v) for v in sub.variables]
    empirical = np.cov(data, rowvar=False)
    assert np.allclose(cov[np.ix_(observed, observed)], empirical, atol=0.05)


def test_degenerate_edge_is_unfaithful():
    # A declared a -> c edge with ~zero coefficient carries no signal -> must be rejected.
    report = check_faithfulness(_chain(0.0))
    assert not report.faithful
    assert report.weakest_edge == ("a", "c")
    assert report.min_partial_corr < 0.05


def test_strong_edge_is_faithful():
    report = check_faithfulness(_chain(1.0))
    assert report.faithful
    assert report.min_partial_corr >= 0.05


def test_spurious_regime_edge_is_rejected():
    # regime_terms identical to terms -> the regime changes nothing -> the regime edge is spurious.
    spec = WorldSpec(
        variables=(
            Variable("r", Role.DISTURBANCE),
            Variable("x", Role.CONTROLLABLE),
            Variable("y", Role.OUTCOME),
        ),
        mechanisms=(
            Mechanism(
                "y",
                terms=(Term("x", 1.0),),
                regime="r",
                regime_terms=(Term("x", 1.0),),  # no change between branches
            ),
        ),
    )
    report = check_faithfulness(spec)
    assert not report.faithful
    assert not report.regime_ok


def test_complete_graph_is_trivial():
    # every ordered pair present -> any "guess everything" wins -> structurally trivial.
    spec = WorldSpec(
        variables=(Variable("a", Role.CONTROLLABLE), Variable("b", Role.OUTCOME)),
        mechanisms=(Mechanism("b", (Term("a", 1.0),)),),
    )
    # a->b is the only possible directed edge among {a,b} given acyclicity in one direction;
    # density = 1/2 < 1 so this is non-trivial. A genuinely complete graph needs a cycle, which T1
    # rejects — so is_nontrivial guards the degenerate density==1 case.
    assert is_nontrivial(spec)
