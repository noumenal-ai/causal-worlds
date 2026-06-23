"""Name anonymization — the Caliper-style control for the anti-cliché claim.

A causal benchmark that an LLM can solve from *variable names alone* tests memorized priors, not
discovery (Caliper; "Causal Parrots"). The honest control is to **anonymize the names** — relabel
every variable ``X1, X2, …`` while preserving roles, mechanisms, and structure exactly — and show
the name-only baseline then collapses to chance. The gap between the named and anonymized guess is
how much a world leaks through its names; on an anti-cliché world it should be small.

The transform is pure and bijective, so a score computed on the anonymized spec equals the score on
the original under the relabeling — no need to map edges back.
"""

from causal_worlds.schema import Mechanism, Term, Variable, WorldSpec

_PREFIX = "X"


def _rename_term(term: Term, mapping: dict[str, str]) -> Term:
    """A term with its parent relabeled (coefficient and lag unchanged)."""
    return Term(parent=mapping[term.parent], coeff=term.coeff, lag=term.lag)


def _rename_mechanism(mechanism: Mechanism, mapping: dict[str, str]) -> Mechanism:
    """A mechanism with its target, parents, and regime switch relabeled."""
    return Mechanism(
        target=mapping[mechanism.target],
        terms=tuple(_rename_term(t, mapping) for t in mechanism.terms),
        noise_scale=mechanism.noise_scale,
        regime=mapping[mechanism.regime] if mechanism.regime is not None else None,
        regime_terms=(
            tuple(_rename_term(t, mapping) for t in mechanism.regime_terms)
            if mechanism.regime_terms is not None
            else None
        ),
    )


def anonymize_spec(spec: WorldSpec) -> tuple[WorldSpec, dict[str, str]]:
    """Relabel every variable to ``X1, X2, …``; return the anonymized spec and ``{original: anon}``.

    Roles, hidden flags, coefficients, lags, and graph structure are preserved exactly — only the
    semantic names an LLM could pattern-match are stripped (the Caliper anti-cliché control).
    """
    mapping = {v.name: f"{_PREFIX}{i + 1}" for i, v in enumerate(spec.variables)}
    variables = tuple(
        Variable(name=mapping[v.name], role=v.role, hidden=v.hidden) for v in spec.variables
    )
    mechanisms = tuple(_rename_mechanism(m, mapping) for m in spec.mechanisms)
    return WorldSpec(variables=variables, mechanisms=mechanisms), mapping
