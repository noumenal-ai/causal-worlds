"""Integration test: the reference grader recovers a confounder + regime-flip trap world.

This is the project's spike result, pinned as a CI test: standard observational methods fail this
world, but the interventional-CI grader recovers the structure and drops the hidden-confounded edge.
"""

from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.evaluation import score
from causal_worlds.sample import build_substrate
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec, answer_key


def _trap_world():
    """Hidden L confounds overtime~sales; price->demand flips sign by regime R (the trap world)."""
    variables = (
        Variable("R", Role.DISTURBANCE),
        Variable("price", Role.CONTROLLABLE),
        Variable("L", Role.DISTURBANCE, hidden=True),
        Variable("foot", Role.OBSERVABLE),
        Variable("overtime", Role.OBSERVABLE),  # a leaf child of L (+foot): confounded with sales
        Variable("demand", Role.OBSERVABLE),
        Variable("sales", Role.OUTCOME),
    )
    mechanisms = (
        Mechanism("foot", (Term("L", 0.8),)),
        Mechanism("overtime", (Term("L", 0.8), Term("foot", 0.3))),
        Mechanism(
            "demand",
            terms=(Term("price", -1.0), Term("foot", 0.5), Term("R", 2.0)),
            regime="R",
            regime_terms=(Term("price", 1.0), Term("foot", 0.5), Term("R", 2.0)),
        ),
        Mechanism("sales", (Term("demand", 1.0), Term("foot", 0.4), Term("L", 0.6))),
    )
    return WorldSpec(variables=variables, mechanisms=mechanisms)


def test_grader_recovers_trap_world():
    spec = _trap_world()
    substrate = build_substrate(spec)
    recovered = InterventionalCiDiscoverer(n=4000).recover(substrate, seed=7)
    report = score(recovered, answer_key(spec))
    assert report.directed_shd <= 1  # near-perfect recovery (validated in the spikes)
    assert report.confounded_reported == 0  # the hidden-confounded overtime~sales edge is dropped
