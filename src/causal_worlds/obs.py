"""The observability/tracing seam.

The package depends only on :class:`Tracer`, never on a concrete backend. :class:`NullTracer` is the
default (records nothing); :class:`LangfuseTracer` sends spans to Langfuse (OpenTelemetry) when the
``observability`` extra is installed and Langfuse keys are in the environment — turned on via
``CAUSAL_WORLDS_LANGFUSE_ENABLED=true``, so a misconfigured backend can never break a run.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langfuse import Langfuse


@runtime_checkable
class Tracer(Protocol):
    """Wrap a unit of work in an observability span."""

    def span(self, name: str, **metadata: object) -> AbstractContextManager[None]:
        """Open a span named ``name`` (with optional metadata) around a unit of work."""
        ...

    def flush(self) -> None:
        """Send any buffered spans — call before a short-lived process exits."""
        ...


class NullTracer:
    """A tracer that records nothing — the default until observability is configured."""

    @contextmanager
    def span(self, name: str, **metadata: object) -> Iterator[None]:
        """A no-op span (name and metadata are ignored)."""
        _ = (name, metadata)
        yield

    def flush(self) -> None:
        """No-op flush."""


class LangfuseTracer:
    """A :class:`Tracer` that emits spans to Langfuse (OpenTelemetry) via an injected client."""

    def __init__(self, client: "Langfuse") -> None:
        """Store the Langfuse client (constructed at the edge by :func:`build_langfuse_tracer`)."""
        self._client = client

    @contextmanager
    def span(self, name: str, **metadata: object) -> Iterator[None]:
        """Open a Langfuse span named ``name`` (attaching ``metadata`` if any) for the block."""
        with self._client.start_as_current_observation(name=name, as_type="span"):
            if metadata:
                self._client.update_current_span(metadata=metadata)
            yield

    def flush(self) -> None:
        """Flush buffered spans to Langfuse (best practice before a script/CLI exits)."""
        self._client.flush()


def build_langfuse_tracer() -> LangfuseTracer:  # pragma: no cover - real backend wiring
    """Construct a live Langfuse tracer; needs the ``observability`` extra + Langfuse keys in env.

    The client reads ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST`` itself.
    """
    from langfuse import Langfuse  # noqa: PLC0415 - lazy: Langfuse is an optional extra

    return LangfuseTracer(Langfuse())
