"""Tests for grading a discoverer against a world (the package's core use)."""

from pathlib import Path

from typer.testing import CliRunner

from causal_worlds import answer_key, worlds
from causal_worlds.artifact import load_bundle
from causal_worlds.bench import grade_bundle, grade_spec
from causal_worlds.cli import app

runner = CliRunner()
_BUNDLE = Path("benchmark/v0.2/world_00")  # a committed bundle


class _PerfectDiscoverer:
    """Returns a fixed edge set — for testing the grading path without real discovery."""

    def __init__(self, edges):
        self._edges = edges

    def recover(self, substrate, *, seed):  # noqa: ARG002
        return self._edges


def test_grade_spec_scores_a_discoverer():
    coffee = worlds.get("coffee")
    report = grade_spec(coffee, _PerfectDiscoverer(answer_key(coffee).edges))
    assert report.directed_shd == 0
    assert report.f1 == 1.0


def test_grade_bundle_loads_and_scores():
    spec = load_bundle(_BUNDLE).spec  # grade a perfect discoverer against the bundle's own truth
    report = grade_bundle(_BUNDLE, _PerfectDiscoverer(answer_key(spec).edges))
    assert report.directed_shd == 0
    assert report.f1 == 1.0


def test_score_cli_with_default_reference_discoverer():
    result = runner.invoke(app, ["score", str(_BUNDLE), "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "directed_shd=" in result.stdout


def test_score_cli_with_dynamic_discoverer():
    result = runner.invoke(
        app,
        [
            "score",
            str(_BUNDLE),
            "--discoverer",
            "causal_worlds.discover:InterventionalCiDiscoverer",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "f1=" in result.stdout


def test_score_cli_rejects_bad_discoverer_path():
    result = runner.invoke(app, ["score", str(_BUNDLE), "--discoverer", "no_colon_here"])
    assert result.exit_code == 1
