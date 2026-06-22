"""The causal-worlds command-line interface (the construction-from-use edge)."""

import typer

from causal_worlds import __version__

app = typer.Typer(
    help="Generate and grade fictional causal worlds.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Generate and grade fictional causal worlds."""


@app.command()
def version() -> None:
    """Print the installed causal-worlds version."""
    typer.echo(__version__)
