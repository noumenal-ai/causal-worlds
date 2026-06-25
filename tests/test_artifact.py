"""Tests for admitted-world bundle persistence."""

import json

from causal_worlds import worlds
from causal_worlds.artifact import Provenance, load_bundle, save_bundle
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.fakes import FakeAuthor, FakeJudge
from causal_worlds.generate import generate

_FAST = InterventionalCiDiscoverer(n=4000)


def _admit_coffee():
    return generate(
        "a coffee chain",
        author=FakeAuthor([worlds.get("coffee")]),
        discoverer=_FAST,
        judge=FakeJudge(),
        seed=7,
    )


def test_bundle_round_trips(tmp_path):
    world = _admit_coffee()
    provenance = Provenance(
        author_model="fake",
        judge_model="fake",
        grader="InterventionalCiDiscoverer",
        grader_version="1",
        seed=7,
        n_rows=200,
        created_at="2026-06-23T00:00:00",
    )
    directory = save_bundle(world, tmp_path / "coffee", provenance=provenance)
    loaded = load_bundle(directory)

    assert loaded.spec == world.spec
    assert "local_buzz" not in loaded.columns  # hidden confounder is never emitted as data
    assert loaded.data.shape == (200, len(loaded.columns))
    assert loaded.manifest["prompt"] == "a coffee chain"
    assert loaded.manifest["author_model"] == "fake"
    assert loaded.manifest["difficulty"] == 1.0
    assert loaded.manifest["grade"]["directed_shd"] == 0


def test_answer_key_file_lists_the_confounded_pair(tmp_path):
    world = _admit_coffee()
    provenance = Provenance(author_model="fake", grader="g", grader_version="1", seed=7, n_rows=100)
    directory = save_bundle(world, tmp_path / "c", provenance=provenance)
    key = json.loads((directory / "answer_key.json").read_text())
    assert ["overtime", "sales"] in key["confounded"]
