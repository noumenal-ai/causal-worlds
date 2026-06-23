"""Benchmark a causal-discovery method against a shipped world — the package's reason to exist.

Point any :class:`Discoverer` at a persisted bundle (or a built-in :class:`WorldSpec`) and get a
:class:`Report` scoring it against the declared answer-key. The discoverer drives its own sampling
from the rebuilt substrate, so this works for observational and interventional methods alike.
"""

from pathlib import Path

from causal_worlds.artifact import load_bundle
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.evaluation import Report, TemporalReport, score, temporal_score
from causal_worlds.protocols import Discoverer, TemporalDiscoverer
from causal_worlds.sample import build_substrate
from causal_worlds.schema import WorldSpec, answer_key, temporal_answer_key


def grade_spec(spec: WorldSpec, discoverer: Discoverer | None = None, *, seed: int = 0) -> Report:
    """Grade a discoverer on a :class:`WorldSpec` against its derived answer-key.

    Args:
        spec: The world to grade against (its answer-key is the ground truth).
        discoverer: The method under test (defaults to the reference interventional-CI grader).
        seed: Seeds the discoverer's sampling.

    Returns:
        The :class:`Report` (directed/skeleton SHD, F1, confounded_reported).
    """
    grader = discoverer if discoverer is not None else InterventionalCiDiscoverer()
    recovered = grader.recover(build_substrate(spec), seed=seed)
    return score(recovered, answer_key(spec))


def grade_bundle(
    bundle_dir: Path, discoverer: Discoverer | None = None, *, seed: int = 0
) -> Report:
    """Grade a discoverer on a persisted world bundle (see :func:`grade_spec`)."""
    return grade_spec(load_bundle(bundle_dir).spec, discoverer, seed=seed)


def grade_temporal_spec(
    spec: WorldSpec, discoverer: TemporalDiscoverer, *, seed: int = 0
) -> TemporalReport:
    """Grade a time-series discoverer on a (temporal) spec against its lagged answer-key.

    Args:
        spec: The temporal world to grade against.
        discoverer: The time-series method under test (recovers ``(src, dst, lag)`` edges).
        seed: Seeds the discoverer's sampling.

    Returns:
        The :class:`TemporalReport` (temporal SHD + F1 over lagged edges).
    """
    recovered = discoverer.recover_temporal(build_substrate(spec), seed=seed)
    return temporal_score(recovered, temporal_answer_key(spec))
