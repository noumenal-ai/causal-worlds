"""The observability/tracing seam.

v0.1 ships a no-op tracer; a Langfuse (OpenTelemetry) adapter lands in v0.2 behind this same seam —
the rest of the package depends only on :class:`Tracer`, never on a concrete backend.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tracer(Protocol):
    """Wrap a unit of work in an observability span."""

    def span(self, name: str) -> AbstractContextManager[None]:
        """Open a span named ``name``; use as a context manager around a unit of work."""
        ...


class NullTracer:
    """A tracer that records nothing — the default until observability is configured."""

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """A no-op span (the name is ignored)."""
        _ = name
        yield
