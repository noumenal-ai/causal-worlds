"""Structural difficulty — how hard a world is to *discover*, read off the spec, not the names.

v0.3's crossover surfaced an honest negative: the anti-cliché difficulty score (how guessable the
graph is from variable *names*) did not predict how badly the standard methods failed. The hardness
lives in the *structure* — hidden confounders and regime sign-flips — which name-guessability can't
see. This module scores that structure directly, from the spec and its derived answer-key:

* ``hidden_confounders`` — latent variables that are a common cause of two or more observed vars.
* ``confounded_pairs`` — observed pairs sharing a hidden cause with no direct edge (the trap that
  defeats methods assuming causal sufficiency).
* ``sign_flips`` — (mechanism, parent) pairs whose coefficient changes sign between regimes (the
  effect reverses — linear methods average it toward zero).
* ``density`` — observed edges over the possible ordered pairs.

The headline ``score`` is the trap count (confounded pairs + sign-flips): the two features that make
a world hard for observational/score-based discovery specifically. Pure and deterministic.
"""

from dataclasses import dataclass

from causal_worlds.schema import Mechanism, WorldSpec, answer_key


@dataclass(frozen=True, slots=True)
class StructuralDifficulty:
    """Discovery-hardness read off a world's structure (higher = harder for standard methods)."""

    score: float
    hidden_confounders: int
    confounded_pairs: int
    sign_flips: int
    density: float


def _sign_flips(mechanism: Mechanism) -> int:
    """Count parents whose coefficient flips sign between the default and regime branches."""
    if mechanism.regime_terms is None:
        return 0
    default = {term.parent: term.coeff for term in mechanism.terms}
    flips = 0
    for term in mechanism.regime_terms:
        base = default.get(term.parent)
        if base is not None and base * term.coeff < 0:
            flips += 1
    return flips


def _hidden_confounders(spec: WorldSpec) -> int:
    """Count hidden variables that directly cause two or more observed variables."""
    observed = {v.name for v in spec.variables if not v.hidden}
    hidden = {v.name for v in spec.variables if v.hidden}
    children: dict[str, set[str]] = {name: set() for name in hidden}
    for mechanism in spec.mechanisms:
        if mechanism.target not in observed:
            continue
        for term in (*mechanism.terms, *(mechanism.regime_terms or ())):
            if term.parent in hidden:
                children[term.parent].add(mechanism.target)
    return sum(1 for targets in children.values() if len(targets) >= 2)  # noqa: PLR2004


def structural_difficulty(spec: WorldSpec) -> StructuralDifficulty:
    """Score how hard ``spec`` is to discover from its structure (the answer-key + mechanisms)."""
    key = answer_key(spec)
    confounded_pairs = len(key.confounded)
    sign_flips = sum(_sign_flips(m) for m in spec.mechanisms)
    hidden = _hidden_confounders(spec)
    n_observed = sum(1 for v in spec.variables if not v.hidden)
    possible = n_observed * (n_observed - 1)
    density = len(key.edges) / possible if possible else 0.0
    return StructuralDifficulty(
        score=float(confounded_pairs + sign_flips),
        hidden_confounders=hidden,
        confounded_pairs=confounded_pairs,
        sign_flips=sign_flips,
        density=density,
    )
