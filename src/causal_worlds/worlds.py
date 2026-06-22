"""Built-in example worlds — ready-to-run specs, each with a derived ground-truth answer-key.

``coffee`` is the hero: a hidden confounder (L -> overtime, sales) plus a regime sign-flip (price ->
demand by R) — the trap that defeats standard observational discovery. ``ecommerce`` is an easy
textbook control. Both pass :func:`causal_worlds.schema.validate`.
"""

from causal_worlds.errors import CausalWorldsError
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


class UnknownWorldError(CausalWorldsError):
    """No built-in world is registered under the requested name."""


def _coffee() -> WorldSpec:
    """Hidden L confounds overtime~sales; price->demand flips sign by regime R."""
    variables = (
        Variable("R", Role.DISTURBANCE),
        Variable("price", Role.CONTROLLABLE),
        Variable("L", Role.DISTURBANCE, hidden=True),
        Variable("foot", Role.OBSERVABLE),
        Variable("overtime", Role.OBSERVABLE),
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


def _ecommerce() -> WorldSpec:
    """Textbook control: ad spend -> traffic -> sales; discounts -> sales. No hidden confounder."""
    variables = (
        Variable("ad", Role.CONTROLLABLE),
        Variable("discount", Role.CONTROLLABLE),
        Variable("traffic", Role.OBSERVABLE),
        Variable("sales", Role.OUTCOME),
    )
    mechanisms = (
        Mechanism("traffic", (Term("ad", 1.0),)),
        Mechanism("sales", (Term("traffic", 1.0), Term("discount", 0.8))),
    )
    return WorldSpec(variables=variables, mechanisms=mechanisms)


BUILTINS: dict[str, WorldSpec] = {
    "coffee": _coffee(),
    "ecommerce": _ecommerce(),
}


def names() -> list[str]:
    """The names of the built-in worlds, sorted."""
    return sorted(BUILTINS)


def get(name: str) -> WorldSpec:
    """Return a built-in world spec by name.

    Args:
        name: A built-in world name (see :func:`names`).

    Returns:
        The world spec.

    Raises:
        UnknownWorldError: No built-in world has that name.
    """
    try:
        return BUILTINS[name]
    except KeyError:
        msg = f"unknown world {name!r}; available: {', '.join(names())}"
        raise UnknownWorldError(msg) from None
