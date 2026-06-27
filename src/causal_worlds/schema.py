"""The world-spec IR (the generative truth) plus its derived answer-key and static validation.

A :class:`WorldSpec` is the single source of truth: variables (incl. hidden confounders) plus a
generative :class:`Mechanism` per non-root. The answer-key is *derived* by :func:`answer_key`,
never stored separately, so the spec and the key can never disagree.

:func:`validate` is the static T1 gate: it rejects ill-formed specs before any sampling happens.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import overload

import numpy as np
from numpy.typing import NDArray

from causal_worlds.errors import CausalWorldsError

type _FloatArray = NDArray[np.float64]
type _TransformFn = Callable[[float | _FloatArray], float | _FloatArray]


class Role(StrEnum):
    """The role a variable plays in an operation."""

    CONTROLLABLE = "controllable"
    OBSERVABLE = "observable"
    DISTURBANCE = "disturbance"
    OUTCOME = "outcome"


class Transform(StrEnum):
    """A nonlinearity applied to a parent *before* its coefficient: ``coeff·f(parent)``.

    A mechanism is then a **generalized additive model**, ``X = Σ coeffᵢ·fᵢ(parentᵢ) + noise`` — the
    *additive nonlinear* form (issue #10). The noise stays additive, which is what keeps abduction
    closed-form (``noise = factual - f(parents)``), so counterfactuals remain exact; and the
    *answer-key edge set is unchanged* (a transform alters how a parent acts, not whether it does),
    so grading stays correct-by-construction.

    Every member is **total over the reals** (safe on standardized, possibly-negative data — no
    domain errors, no NaNs) and serializes to a string, so a world stays a pure declarative spec.
    ``IDENTITY`` recovers the linear-Gaussian mechanism, so existing worlds are unchanged.
    """

    IDENTITY = "identity"
    SQUARE = "square"
    CUBE = "cube"
    TANH = "tanh"
    RELU = "relu"
    ABS = "abs"

    @overload
    def apply(self, x: float) -> float: ...
    @overload
    def apply(self, x: _FloatArray) -> _FloatArray: ...
    def apply(self, x: float | _FloatArray) -> float | _FloatArray:
        """Apply the nonlinearity elementwise (works on a scalar or a NumPy column)."""
        return _TRANSFORMS[self](x)


_TRANSFORMS: dict[Transform, _TransformFn] = {
    Transform.IDENTITY: lambda x: x,
    Transform.SQUARE: np.square,
    Transform.CUBE: lambda x: np.power(x, 3.0),
    Transform.TANH: np.tanh,
    Transform.RELU: lambda x: np.maximum(x, 0.0),
    Transform.ABS: np.abs,
}


@dataclass(frozen=True, slots=True)
class Variable:
    """A variable in a world. Hidden variables drive sampling but are not emitted as data."""

    name: str
    role: Role
    hidden: bool = False


@dataclass(frozen=True, slots=True)
class Term:
    """A single term ``coeff · transform(parent)`` in a mechanism.

    With the default ``transform`` (``IDENTITY``) this is the linear term ``coeff·parent``; a
    non-identity :class:`Transform` (e.g. ``SQUARE``) makes the term *additive nonlinear*, so a
    mechanism becomes a generalized additive model ``Σ coeffᵢ·fᵢ(parentᵢ) + noise``.

    ``lag`` is how many timesteps back the parent is read: ``0`` is contemporaneous (the default;
    purely cross-sectional worlds use only this), ``>= 1`` is a temporal/lagged edge. A lagged
    self-reference (``parent`` equals the target) is autoregression and is allowed; a *lag-0*
    self-reference is an instantaneous cycle and is rejected.
    """

    parent: str
    coeff: float
    lag: int = 0
    transform: Transform = Transform.IDENTITY


@dataclass(frozen=True, slots=True)
class Mechanism:
    """How one variable is generated: an additive function of its parents plus Gaussian noise.

    Each term contributes ``coeff·transform(parent)``; with all-``IDENTITY`` transforms this is the
    linear-Gaussian mechanism, and a non-identity transform makes it *additive nonlinear*. A
    variable with no terms and no regime is an exogenous root (pure noise). When ``regime`` names a
    binary variable, ``regime_terms`` apply on rows where it is truthy and ``terms`` on the rest —
    how a regime flips or rescales an effect (the anti-cliché lever).
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


def _contemporaneous_parents(mechanism: Mechanism) -> set[str]:
    """Parents read at the *current* timestep (lag-0 terms + the regime switch).

    These are the only edges that constrain within-timestep evaluation order and the only ones that
    can form an instantaneous cycle; lagged edges read already-computed history.
    """
    parents = {term.parent for term in mechanism.terms if term.lag == 0}
    parents |= {term.parent for term in (mechanism.regime_terms or ()) if term.lag == 0}
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
    """Raise :class:`CyclicGraphError` if the *contemporaneous* graph has a cycle.

    Only lag-0 edges are checked: lagged edges read past timesteps, so a temporal cycle (e.g. an
    autoregressive self-loop, or ``x_t -> y_{t+1} -> x_{t+2}``) is legitimate, not a contradiction.
    """
    adjacency: dict[str, list[str]] = {variable.name: [] for variable in spec.variables}
    for mechanism in spec.mechanisms:
        for parent in _contemporaneous_parents(mechanism):
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

    # The summary graph: collapse lags, drop autoregressive self-loops (not cross-variable edges).
    edges = {
        (parent, mechanism.target)
        for mechanism in spec.mechanisms
        for parent in _parents(mechanism)
        if parent in observed and mechanism.target in observed and parent != mechanism.target
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


def temporal_answer_key(spec: WorldSpec) -> frozenset[tuple[str, str, int]]:
    """Derive the lagged ground-truth edges ``(src, dst, lag)`` over observed variables.

    Unlike the summary :func:`answer_key`, this keeps the lag on each edge and retains
    autoregressive self-loops (``src == dst``, ``lag >= 1``) — the truth a time-series method is
    graded against. Regime terms contribute their own lagged edges.
    """
    observed = {variable.name for variable in spec.variables if not variable.hidden}
    return frozenset(
        (term.parent, mechanism.target, term.lag)
        for mechanism in spec.mechanisms
        for term in (*mechanism.terms, *(mechanism.regime_terms or ()))
        if term.parent in observed and mechanism.target in observed
    )


def has_nonlinear_terms(spec: WorldSpec) -> bool:
    """Whether any mechanism uses a non-identity :class:`Transform` (an additive-nonlinear world).

    The linear-only machinery (e.g. the closed-form control optimum, which solves ``(I - B)⁻¹``) is
    valid only when this is ``False``; sampling, grading, and counterfactuals handle either.
    """
    return any(
        term.transform is not Transform.IDENTITY
        for mechanism in spec.mechanisms
        for term in (*mechanism.terms, *(mechanism.regime_terms or ()))
    )
