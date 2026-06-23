"""Tests for the SCM substrate (the deterministic functional core)."""

import numpy as np

from causal_worlds.sample import build_substrate
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


def _coffee():
    """Coffee world: hidden L confounds foot & sales; price->demand flips sign by regime R."""
    variables = (
        Variable("R", Role.DISTURBANCE),
        Variable("price", Role.CONTROLLABLE),
        Variable("L", Role.DISTURBANCE, hidden=True),
        Variable("foot", Role.OBSERVABLE),
        Variable("demand", Role.OBSERVABLE),
        Variable("sales", Role.OUTCOME),
    )
    mechanisms = (
        Mechanism("foot", (Term("L", 0.8),)),
        Mechanism(
            "demand",
            terms=(Term("price", -1.0), Term("foot", 0.5)),
            regime="R",
            regime_terms=(Term("price", 1.0), Term("foot", 0.5)),
        ),
        Mechanism("sales", (Term("demand", 1.0), Term("foot", 0.4), Term("L", 0.6))),
    )
    return WorldSpec(variables=variables, mechanisms=mechanisms)


def test_deterministic_given_seed():
    sub = build_substrate(_coffee())
    assert np.array_equal(sub.sample(500, seed=7).data, sub.sample(500, seed=7).data)


def test_shape_and_hidden_excluded():
    sub = build_substrate(_coffee())
    s = sub.sample(300, seed=1)
    assert s.data.shape == (300, len(sub.variables))
    assert "L" not in sub.variables
    assert set(sub.variables) == {"R", "price", "foot", "demand", "sales"}


def test_intervention_forces_constant():
    sub = build_substrate(_coffee(), standardize=False)  # raw mechanics
    s = sub.sample(200, seed=3, do={"price": 2.0})
    col = sub.variables.index("price")
    assert np.allclose(s.data[:, col], 2.0)
    assert s.intervened == frozenset({"price"})


def test_intervention_propagates_to_descendant():
    sub = build_substrate(
        _coffee(), standardize=False
    )  # raw mechanics (standardizing centers means)
    hi = sub.sample(2000, seed=5, do={"demand": 10.0})
    lo = sub.sample(2000, seed=5, do={"demand": -10.0})
    sales = sub.variables.index("sales")
    # sales = 1.0*demand + ...  -> forcing demand high vs low must move the sales mean a lot
    assert hi.data[:, sales].mean() > lo.data[:, sales].mean() + 5.0


def test_standardize_zscores_continuous_but_leaves_regimes():
    sub = build_substrate(_coffee())  # standardized by default
    data = sub.sample(3000, seed=1).data
    regime = sub.variables.index("R")  # binary regime — left as-is ({0, 1})
    sales = sub.variables.index("sales")  # continuous — z-scored
    assert set(np.unique(data[:, regime])) <= {0.0, 1.0}
    assert abs(data[:, sales].mean()) < 1e-6
    assert abs(data[:, sales].std() - 1.0) < 1e-6
