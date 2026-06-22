"""Tests for the CLI surface."""

from typer.testing import CliRunner

from causal_worlds import __version__
from causal_worlds.cli import app

runner = CliRunner()


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
