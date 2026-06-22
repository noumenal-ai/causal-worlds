"""The small DI container — the composition root for every interface (CLI, library, future web).

Interfaces depend on the container, not on concrete implementations, so construction stays separate
from use and the CLI is not a privileged entry point.
"""

from dataclasses import dataclass

from causal_worlds.config import Settings
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.obs import NullTracer, Tracer
from causal_worlds.protocols import Discoverer


@dataclass(frozen=True, slots=True)
class Container:
    """Resolves the package's services from settings."""

    settings: Settings

    def discoverer(self) -> Discoverer:
        """The reference causal-discovery grader."""
        return InterventionalCiDiscoverer(n=self.settings.discoverer_n)

    def tracer(self) -> Tracer:
        """The observability tracer (no-op until Langfuse is wired in v0.2)."""
        return NullTracer()


def build_container(settings: Settings | None = None) -> Container:
    """Build the default container, loading :class:`Settings` from the env if not provided."""
    return Container(settings=settings if settings is not None else Settings())
