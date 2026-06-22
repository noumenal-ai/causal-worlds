"""Tests for the CLI surface."""

import json

from typer.testing import CliRunner

from causal_worlds import __version__, worlds
from causal_worlds.cli import app
from causal_worlds.fakes import FakeAuthor, FakeJudge
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec

runner = CliRunner()


def _wire_fakes(monkeypatch, specs):
    """Point the container's live builders at fakes + shrink the grader, so the CLI runs keyless."""
    monkeypatch.setenv("CAUSAL_WORLDS_DISCOVERER_N", "4000")
    monkeypatch.setattr(
        "causal_worlds.container.build_claude_author", lambda _model: FakeAuthor(specs)
    )
    monkeypatch.setattr("causal_worlds.container.build_gemini_judge", lambda _model: FakeJudge())


def _bad_spec():
    return WorldSpec(
        variables=(Variable("a", Role.OBSERVABLE), Variable("b", Role.OUTCOME)),
        mechanisms=(Mechanism("b", (Term("a", 1.0),)),),
    )


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_worlds_lists_builtins():
    result = runner.invoke(app, ["worlds"])
    assert result.exit_code == 0
    assert "coffee" in result.stdout
    assert "ecommerce" in result.stdout


def test_gate_command_admits_ecommerce():
    result = runner.invoke(app, ["gate", "ecommerce", "--seed", "7"])
    assert result.exit_code == 0
    assert "admitted=True" in result.stdout


def test_grade_command_runs():
    result = runner.invoke(app, ["grade", "ecommerce", "--seed", "7"])
    assert result.exit_code == 0
    assert "directed_shd=" in result.stdout


def test_unknown_world_exits_nonzero():
    result = runner.invoke(app, ["gate", "does-not-exist"])
    assert result.exit_code == 1


def test_generate_command_writes_a_bundle(tmp_path, monkeypatch):
    _wire_fakes(monkeypatch, [worlds.get("ecommerce")])
    out = tmp_path / "world"
    result = runner.invoke(app, ["generate", "a webshop", str(out), "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "admitted ->" in result.stdout
    assert (out / "manifest.json").exists()
    assert (out / "data.npz").exists()


def test_generate_command_reports_not_admitted(tmp_path, monkeypatch):
    _wire_fakes(monkeypatch, [_bad_spec()])
    monkeypatch.setenv("CAUSAL_WORLDS_MAX_ATTEMPTS", "1")
    result = runner.invoke(app, ["generate", "junk", str(tmp_path / "w"), "--seed", "7"])
    assert result.exit_code == 1
    assert "not admitted" in result.output


def test_generate_command_without_llm_extra_hints(tmp_path, monkeypatch):
    def _boom(_model):
        msg = "no provider sdk"
        raise ImportError(msg)

    monkeypatch.setattr("causal_worlds.container.build_claude_author", _boom)
    result = runner.invoke(app, ["generate", "x", str(tmp_path / "w")])
    assert result.exit_code == 1
    assert "llm" in result.output


def test_benchmark_command_writes_a_set(tmp_path, monkeypatch):
    _wire_fakes(monkeypatch, [worlds.get("ecommerce")])
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a webshop\n# a comment line\nanother shop\n")
    out = tmp_path / "set"
    result = runner.invoke(app, ["benchmark", str(prompts), str(out), "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "2/2 admitted" in result.stdout
    index = json.loads((out / "index.json").read_text())
    assert len(index) == 2
    assert all(row["admitted"] for row in index)
    assert (out / "world_00" / "manifest.json").exists()
