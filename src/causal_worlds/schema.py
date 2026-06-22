"""The world-spec IR (the generative truth) plus its derived answer-key and static validation.

A :class:`WorldSpec` is the single source of truth: variables (incl. hidden confounders) plus a
generative :class:`Mechanism` per non-root. The answer-key is *derived* by :func:`answer_key`,
never stored separately, so the spec and the key can never disagree.

:func:`validate` is the static T1 gate: it rejects ill-formed specs before any sampling happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from causal_worlds.errors import CausalWorldsError


class Role(StrEnum):
    """The role a variable plays in an operation."""

    CONTROLLABLE = "controllable"
    OBSERVABLE = "observable"
    DISTURBANCE = "disturbance"
    OUTCOME = "outcome"


@dataclass(frozen=True, slots=True)
class Variable:
    """A variable in a world. Hidden variables drive sampling but are not emitted as data."""

    name: str
    role: Role
    hidden: bool = False


@dataclass(frozen=True, slots=True)
class Term:
    """A single linear term ``coeff * parent`` in a mechanism."""

    parent: str
    coeff: float


@dataclass(frozen=True, slots=True)
class Mechanism:
    """How one variable is generated: a linear function of parents plus Gaussian noise.

    A variable with no terms and no regime is an exogenous root (pure noise). When ``regime``
    names a binary variable, ``regime_terms`` apply on rows where it is truthy and ``terms`` on
    the rest — how a regime flips or rescales an effect (the anti-cliché lever).
    """

    target: str
    terms: tuple[Term, ...] = ()
    noise_scale: float = 0.3
    regime: str | None = None
    regime_terms: tuple[Term, ...] | None = None


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """A fictional world: variables (incl. hidden) plus the generative mechanism per non-root."""

    variables: tuple[Variable, ...]
    mechanisms: tuple[Mechanism, ...]


@dataclass(frozen=True, slots=True)
class AnswerKey:
    """The ground truth a discoverer is graded against.

    ``edges`` = directed causal edges over **observed** variables. ``confounded`` = observed
    pairs sharing a hidden common cause with **no** direct edge — reporting a causal edge for
    such a pair is wrong (only interventions distinguish confounding from causation).
    """

    edges: frozenset[tuple[str, str]]
    confounded: frozenset[frozenset[str]]


class SpecError(CausalWorldsError):
    """Base error for an invalid :class:`WorldSpec`."""


class DanglingReferenceError(SpecError):
    """A mechanism references a variable not declared in the spec."""


class CyclicGraphError(SpecError):
    """The declared causal graph contains a cycle."""


class RoleError(SpecError):
    """The spec lacks a required role (an observed controllable lever and an observed outcome)."""


class DuplicateMechanismError(SpecError):
    """More than one mechanism targets the same variable."""


def _parents(mechanism: Mechanism) -> set[str]:
    """Return every variable that directly drives a mechanism's target (incl. the regime switch)."""
    parents = {term.parent for term in mechanism.terms}
    parents |= {term.parent for term in (mechanism.regime_terms or ())}
    if mechanism.regime is not None:
        parents.add(mechanism.regime)
    return parents


def validate(spec: WorldSpec) -> None:
    """Validate a world spec — the static T1 gate.

    Args:
        spec: The world spec to check.

    Raises:
        DuplicateMechanismError: Two mechanisms target the same variable.
        DanglingReferenceError: A mechanism references an undeclared variable.
        RoleError: No observed controllable variable, or no observed outcome.
        CyclicGraphError: The declared causal graph is cyclic.
    """
    names = {variable.name for variable in spec.variables}

    seen: set[str] = set()
    for mechanism in spec.mechanisms:
        if mechanism.target not in names:
            msg = f"mechanism targets undeclared variable {mechanism.target!r}"
            raise DanglingReferenceError(msg)
        if mechanism.target in seen:
            msg = f"more than one mechanism targets {mechanism.target!r}"
            raise DuplicateMechanismError(msg)
        seen.add(mechanism.target)
        for parent in _parents(mechanism):
            if parent not in names:
                msg = (
                    f"mechanism for {mechanism.target!r} references undeclared variable {parent!r}"
                )
                raise DanglingReferenceError(msg)

    observed_roles = {variable.role for variable in spec.variables if not variable.hidden}
    if Role.CONTROLLABLE not in observed_roles:
        msg = "a world needs at least one observed controllable variable"
        raise RoleError(msg)
    if Role.OUTCOME not in observed_roles:
        msg = "a world needs at least one observed outcome variable"
        raise RoleError(msg)

    _ensure_acyclic(spec)


def _ensure_acyclic(spec: WorldSpec) -> None:
    """Raise :class:`CyclicGraphError` if the generative graph (parents -> target) has a cycle."""
    adjacency: dict[str, list[str]] = {variable.name: [] for variable in spec.variables}
    for mechanism in spec.mechanisms:
        for parent in _parents(mechanism):
            adjacency[parent].append(mechanism.target)

    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node: str) -> None:
        """Depth-first visit, raising on a back-edge into the current path."""
        visiting.add(node)
        for successor in adjacency[node]:
            if successor in visiting:
                msg = f"cycle through {node}->{successor}"
                raise CyclicGraphError(msg)
            if successor not in done:
                walk(successor)
        visiting.discard(node)
        done.add(node)

    for variable in spec.variables:
        if variable.name not in done:
            walk(variable.name)


def answer_key(spec: WorldSpec) -> AnswerKey:
    """Derive the ground-truth answer-key (observed edges + confounded pairs) from a spec.

    Args:
        spec: A spec that has passed :func:`validate`.

    Returns:
        The directed edges over observed variables and the hidden-confounded observed pairs.
    """
    observed = {variable.name for variable in spec.variables if not variable.hidden}
    hidden = {variable.name for variable in spec.variables if variable.hidden}

    edges = {
        (parent, mechanism.target)
        for mechanism in spec.mechanisms
        for parent in _parents(mechanism)
        if parent in observed and mechanism.target in observed
    }

    children_of: dict[str, set[str]] = {name: set() for name in hidden}
    for mechanism in spec.mechanisms:
        if mechanism.target not in observed:
            continue
        for parent in _parents(mechanism):
            if parent in hidden:
                children_of[parent].add(mechanism.target)

    confounded: set[frozenset[str]] = set()
    for shared_children in children_of.values():
        ordered = sorted(shared_children)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if (left, right) not in edges and (right, left) not in edges:
                    confounded.add(frozenset((left, right)))

    return AnswerKey(edges=frozenset(edges), confounded=frozenset(confounded))
