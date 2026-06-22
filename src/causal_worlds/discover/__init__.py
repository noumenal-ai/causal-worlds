"""The reference causal-discovery grader (interventional-CI)."""

from causal_worlds.discover.interventional import InterventionalCiDiscoverer

GRADER = "interventional-ci"
GRADER_VERSION = "1"  # bump when the grader's recovery behavior changes (provenance in manifests)

__all__ = ["GRADER", "GRADER_VERSION", "InterventionalCiDiscoverer"]
