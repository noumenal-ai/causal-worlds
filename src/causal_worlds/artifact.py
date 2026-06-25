"""Persist an admitted world as a self-describing on-disk bundle, and load it back.

A bundle is a directory of four files:

* ``spec.json`` — the :class:`WorldSpec` (the generative truth), via the pydantic boundary model.
* ``data.npz`` — a sampled time-series of the *observed* variables (the discoverer's input).
* ``answer_key.json`` — the derived ground-truth edges + confounded pairs (from the spec).
* ``manifest.json`` — provenance binding it together: prompt, models, grader version, seed, grade,
  difficulty, and the honesty label (these worlds are fictional, not real-world advice).

Provenance is explicit and complete so a shipped benchmark is reproducible and auditable. The wall
clock is injected (``Provenance.created_at``), not read here, keeping this module deterministic.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_worlds._version import __version__
from causal_worlds.difficulty import structural_difficulty
from causal_worlds.generate import AdmittedWorld
from causal_worlds.sample import build_substrate
from causal_worlds.schema import WorldSpec, answer_key
from causal_worlds.serde import spec_from_json, spec_to_json

_SCHEMA_VERSION = "1"
_HONESTY = "Fictional world for benchmarking causal discovery; not a model of any real system."
_SPEC_FILE = "spec.json"
_DATA_FILE = "data.npz"
_KEY_FILE = "answer_key.json"
_MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where an admitted world came from — recorded in the manifest for reproducibility."""

    author_model: str
    grader: str
    grader_version: str
    seed: int
    n_rows: int
    judge_model: str | None = None
    created_at: str | None = None
    complexity: str | None = None  # the author's requested structural-complexity level, if any
    anti_cliche: bool = (
        True  # whether the anti-cliché gate rejected (benchmark) or was advisory (playground)
    )


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    """A bundle read back from disk."""

    spec: WorldSpec
    columns: tuple[str, ...]
    data: np.ndarray
    manifest: dict[str, object]


def _answer_key_json(spec: WorldSpec) -> dict[str, object]:
    """Serialize the derived answer-key to plain JSON (sorted edges and confounded pairs)."""
    key = answer_key(spec)
    return {
        "edges": sorted([list(edge) for edge in key.edges]),
        "confounded": sorted([sorted(pair) for pair in key.confounded]),
    }


def _manifest(
    world: AdmittedWorld, provenance: Provenance, columns: tuple[str, ...]
) -> dict[str, object]:
    """Assemble the manifest dict from the admitted world and its provenance."""
    report = world.report
    return {
        "schema_version": _SCHEMA_VERSION,
        "package_version": __version__,
        "prompt": world.prompt,
        "created_at": provenance.created_at,
        "honesty": _HONESTY,
        "seed": provenance.seed,
        "n_rows": provenance.n_rows,
        "attempts": world.attempts,
        "author_model": provenance.author_model,
        "judge_model": provenance.judge_model,
        "complexity": provenance.complexity,
        "anti_cliche": provenance.anti_cliche,
        "grader": provenance.grader,
        "grader_version": provenance.grader_version,
        "difficulty": report.difficulty,
        "faithfulness": report.faithfulness,
        "structural_difficulty": asdict(structural_difficulty(world.spec)),
        "grade": asdict(report.grade) if report.grade is not None else None,
        "variables": list(columns),
    }


def save_bundle(world: AdmittedWorld, directory: Path, *, provenance: Provenance) -> Path:
    """Sample the world and write its bundle to ``directory`` (created if absent).

    Args:
        world: The admitted world to persist.
        directory: Target directory for the four bundle files.
        provenance: Reproducibility metadata, including the seed and row count to sample.

    Returns:
        The bundle directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    substrate = build_substrate(world.spec)
    sample = substrate.sample(provenance.n_rows, seed=provenance.seed)
    columns = substrate.variables

    (directory / _SPEC_FILE).write_text(spec_to_json(world.spec))
    np.savez_compressed(directory / _DATA_FILE, data=sample.data, columns=np.array(columns))
    (directory / _KEY_FILE).write_text(json.dumps(_answer_key_json(world.spec), indent=2))
    manifest = _manifest(world, provenance, columns)
    (directory / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return directory


def load_bundle(directory: Path) -> LoadedBundle:
    """Read a bundle written by :func:`save_bundle` back into memory."""
    spec = spec_from_json((directory / _SPEC_FILE).read_text())
    manifest = json.loads((directory / _MANIFEST_FILE).read_text())
    with np.load(directory / _DATA_FILE, allow_pickle=False) as npz:
        data = npz["data"]
        columns = tuple(str(name) for name in npz["columns"])
    return LoadedBundle(spec=spec, columns=columns, data=data, manifest=manifest)
