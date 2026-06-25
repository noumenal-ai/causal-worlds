"""Built-in example worlds — ready-to-run specs, each with a derived ground-truth answer-key.

``coffee`` is the hero: a hidden confounder (``local_buzz`` -> footfall, overtime, sales) plus a
regime sign-flip (``price`` -> demand, inverted on ``weekend``) — the trap that defeats standard
observational discovery. ``ecommerce`` is an easy textbook control. Variable names are deliberately
self-explanatory so the rendered graph reads on its own. Both pass
:func:`causal_worlds.schema.validate`.
"""

from causal_worlds.errors import CausalWorldsError
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


class UnknownWorldError(CausalWorldsError):
    """No built-in world is registered under the requested name."""


def _coffee() -> WorldSpec:
    """Hidden ``local_buzz`` confounds ``overtime ~ sales``; ``price`` flips sign on ``weekend``.

    Names are deliberately self-explanatory (this is the showcase world): ``local_buzz`` is the
    unobserved local activity that lifts footfall, staff overtime, AND sales together — the latent
    confounder a discovery method never sees; ``weekend`` is the regime that raises demand and
    *inverts* price sensitivity (on weekends a higher price no longer deters).
    """
    variables = (
        Variable("weekend", Role.DISTURBANCE),
        Variable("price", Role.CONTROLLABLE),
        Variable("local_buzz", Role.DISTURBANCE, hidden=True),
        Variable("footfall", Role.OBSERVABLE),
        Variable("overtime", Role.OBSERVABLE),
        Variable("demand", Role.OBSERVABLE),
        Variable("sales", Role.OUTCOME),
    )
    mechanisms = (
        Mechanism("footfall", (Term("local_buzz", 0.8),)),
        Mechanism("overtime", (Term("local_buzz", 0.8), Term("footfall", 0.3))),
        Mechanism(
            "demand",
            terms=(Term("price", -1.0), Term("footfall", 0.5), Term("weekend", 2.0)),
            regime="weekend",
            regime_terms=(Term("price", 1.0), Term("footfall", 0.5), Term("weekend", 2.0)),
        ),
        Mechanism("sales", (Term("demand", 1.0), Term("footfall", 0.4), Term("local_buzz", 0.6))),
    )
    return WorldSpec(variables=variables, mechanisms=mechanisms)


def _supply() -> WorldSpec:
    """A temporal supply operation: autoregressive lead time + inventory + a hidden confounder.

    Lags make it genuinely temporal: ``leadtime`` and ``inventory`` carry their own past (AR < 1,
    so stationary), orders arrive next step, and a long lead time depletes next-step inventory. A
    hidden ``logistics`` drives both ``leadtime`` and ``cost`` with no direct edge between them
    (the confounded pair). Grading this needs a time-series method (later); here it exercises the
    temporal substrate and the lagged answer-key.
    """
    variables = (
        Variable("order", Role.CONTROLLABLE),
        Variable("demand", Role.DISTURBANCE),
        Variable("logistics", Role.DISTURBANCE, hidden=True),
        Variable("leadtime", Role.OBSERVABLE),
        Variable("inventory", Role.OBSERVABLE),
        Variable("cost", Role.OBSERVABLE),
        Variable("stockout", Role.OUTCOME),
    )
    mechanisms = (
        Mechanism("leadtime", (Term("logistics", 0.8), Term("leadtime", 0.4, lag=1))),
        Mechanism("cost", (Term("logistics", 0.6), Term("order", 0.3))),
        Mechanism(
            "inventory",
            (
                Term("inventory", 0.5, lag=1),
                Term("order", 0.8, lag=1),
                Term("leadtime", -0.5, lag=1),
            ),
        ),
        Mechanism("stockout", (Term("inventory", -0.8), Term("demand", 0.6))),
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
"""Cross-sectional built-ins — gradeable by the (contemporaneous) reference discoverer today."""

TEMPORAL: dict[str, WorldSpec] = {
    "supply": _supply(),
}
"""Temporal (lagged) built-ins — sampleable now; time-series grading lands in a later release."""

_ALL = {**BUILTINS, **TEMPORAL}


def names() -> list[str]:
    """The cross-sectional built-in world names, sorted (what the CLI grades/gates)."""
    return sorted(BUILTINS)


def temporal_names() -> list[str]:
    """The temporal built-in world names, sorted."""
    return sorted(TEMPORAL)


def get(name: str) -> WorldSpec:
    """Return a built-in world spec by name (cross-sectional or temporal).

    Args:
        name: A built-in world name (see :func:`names` and :func:`temporal_names`).

    Returns:
        The world spec.

    Raises:
        UnknownWorldError: No built-in world has that name.
    """
    try:
        return _ALL[name]
    except KeyError:
        available = ", ".join(sorted(_ALL))
        msg = f"unknown world {name!r}; available: {available}"
        raise UnknownWorldError(msg) from None
