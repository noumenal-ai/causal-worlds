"""causal-worlds: fictional causal operations worlds with a ground-truth answer-key."""

from causal_worlds.errors import BudgetExceededError, CausalWorldsError
from causal_worlds.evaluation import Report, directed_shd, f1, score, skeleton_shd
from causal_worlds.protocols import Discoverer, Edges, Gate, Judge, Substrate
from causal_worlds.sample import Sample, ScmSubstrate, build_substrate
from causal_worlds.schema import (
    AnswerKey,
    CyclicGraphError,
    DanglingReferenceError,
    DuplicateMechanismError,
    Mechanism,
    Role,
    RoleError,
    SpecError,
    Term,
    Variable,
    WorldSpec,
    answer_key,
    validate,
)

__version__ = "0.0.1"

__all__ = [
    "AnswerKey",
    "BudgetExceededError",
    "CausalWorldsError",
    "CyclicGraphError",
    "DanglingReferenceError",
    "Discoverer",
    "DuplicateMechanismError",
    "Edges",
    "Gate",
    "Judge",
    "Mechanism",
    "Report",
    "Role",
    "RoleError",
    "Sample",
    "ScmSubstrate",
    "SpecError",
    "Substrate",
    "Term",
    "Variable",
    "WorldSpec",
    "__version__",
    "answer_key",
    "build_substrate",
    "directed_shd",
    "f1",
    "score",
    "skeleton_shd",
    "validate",
]
