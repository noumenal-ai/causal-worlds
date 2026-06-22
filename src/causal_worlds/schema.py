"""The world-spec / answer-key IR and its static validation (the T1 gate).

A :class:`WorldSpec` is both the build input (what the substrate is compiled from) and, once frozen,
the ground-truth answer-key a discoverer is scored against. :func:`validate` is the static gate: it
rejects specs that are ill-formed before any sampling happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """The role a variable plays in an operation."""

    CONTROLLABLE = "controllable"
    OBSERVABLE = "observable"
    DISTURBANCE = "disturbance"
    OUTCOME = "outcome"


@dataclass(frozen=True, slots=True)
class Variable:
    """A single observed variable in a world."""

    name: str
    role: Role


@dataclass(frozen=True, slots=True)
class Edge:
    """A declared directed causal edge ``src -> dst`` with an integer lag in steps."""

    src: str
    dst: str
    lag: int = 0


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """A fictional world's declared structure: variables plus directed causal edges."""

    variables: tuple[Variable, ...]
    edges: tuple[Edge, ...]


class SpecError(Exception):
    """Base error for an invalid :class:`WorldSpec`."""


class DanglingEdgeError(SpecError):
    """An edge references a variable not declared in the spec."""


class CyclicGraphError(SpecError):
    """The declared causal graph contains a cycle."""


class RoleError(SpecError):
    """The spec lacks a required role (a controllable lever and an outcome)."""


def validate(spec: WorldSpec) -> None:
    """Validate a world spec — the static T1 gate.

    Args:
        spec: The world spec to check.

    Raises:
        DanglingEdgeError: An edge references an undeclared variable.
        RoleError: No controllable variable, or no outcome variable.
        CyclicGraphError: The declared causal graph is cyclic.
    """
    names = {variable.name for variable in spec.variables}
    for edge in spec.edges:
        if edge.src not in names or edge.dst not in names:
            msg = f"edge {edge.src}->{edge.dst} references an undeclared variable"
            raise DanglingEdgeError(msg)

    roles = {variable.role for variable in spec.variables}
    if Role.CONTROLLABLE not in roles:
        msg = "a world needs at least one controllable variable"
        raise RoleError(msg)
    if Role.OUTCOME not in roles:
        msg = "a world needs at least one outcome variable"
        raise RoleError(msg)

    _ensure_acyclic(spec)


def _ensure_acyclic(spec: WorldSpec) -> None:
    """Raise :class:`CyclicGraphError` if the spec's directed graph has a cycle."""
    adjacency: dict[str, list[str]] = {variable.name: [] for variable in spec.variables}
    for edge in spec.edges:
        adjacency[edge.src].append(edge.dst)

    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node: str) -> None:
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
