"""The package exception hierarchy.

Every error subclasses :class:`CausalWorldsError`. We **fail loud**: a missing capability or
malformed input raises — we never fabricate a plausible-looking value.
"""


class CausalWorldsError(Exception):
    """Base class for every error raised by causal-worlds."""


class BudgetExceededError(CausalWorldsError):
    """A per-world compute/LLM budget was exhausted before the work completed."""
