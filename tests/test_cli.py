"""Tests for the CLI surface."""

from typer.testing import CliRunner

from causal_worlds import __version__
from causal_worlds.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
