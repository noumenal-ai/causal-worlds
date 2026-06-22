"""causal-worlds: fictional causal operations worlds with a ground-truth answer-key."""

from causal_worlds.protocols import Discoverer, Edges, Gate, Judge, Substrate
from causal_worlds.schema import (
    CyclicGraphError,
    DanglingEdgeError,
    Edge,
    Role,
    RoleError,
    SpecError,
    Variable,
    WorldSpec,
    validate,
)

__version__ = "0.0.1"

__all__ = [
    "CyclicGraphError",
    "DanglingEdgeError",
    "Discoverer",
    "Edge",
    "Edges",
    "Gate",
    "Judge",
    "Role",
    "RoleError",
    "SpecError",
    "Substrate",
    "Variable",
    "WorldSpec",
    "__version__",
    "validate",
]
