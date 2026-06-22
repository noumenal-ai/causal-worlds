"""The small DI container — the composition root for every interface (CLI, library, future web).

Interfaces depend on the container, not on concrete implementations, so construction stays separate
from use and the CLI is not a privileged entry point. The live author/judge builders are referenced
at module level so they can be substituted in tests without touching a network.
"""

from dataclasses import dataclass

from causal_worlds.author import build_claude_author
from causal_worlds.config import Settings
from causal_worlds.discover import GRADER, GRADER_VERSION, InterventionalCiDiscoverer
from causal_worlds.judge import build_gemini_judge
from causal_worlds.obs import NullTracer, Tracer
from causal_worlds.protocols import Author, Discoverer, Judge


@dataclass(frozen=True, slots=True)
class Container:
    """Resolves the package's services from settings."""

    settings: Settings

    def discoverer(self) -> Discoverer:
        """The reference causal-discovery grader."""
        return InterventionalCiDiscoverer(n=self.settings.discoverer_n)

    def author(self) -> Author:
        """The live world author (Claude); needs the ``llm`` extra and an Anthropic key in env."""
        return build_claude_author(self.settings.author_model)

    def judge(self) -> Judge:
        """The live independent judge (Gemini); needs the ``llm`` extra and a Gemini key in env."""
        return build_gemini_judge(self.settings.judge_model)

    def grader_provenance(self) -> tuple[str, str]:
        """The reference grader's name and version, for the artifact manifest."""
        return GRADER, GRADER_VERSION

    def tracer(self) -> Tracer:
        """The observability tracer (no-op until Langfuse is wired)."""
        return NullTracer()


def build_container(settings: Settings | None = None) -> Container:
    """Build the default container, loading :class:`Settings` from the env if not provided."""
    return Container(settings=settings if settings is not None else Settings())
