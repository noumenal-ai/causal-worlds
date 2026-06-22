"""The causal-worlds command-line interface (the construction-from-use edge)."""

import typer

from causal_worlds import __version__, worlds
from causal_worlds.container import build_container
from causal_worlds.evaluation import score
from causal_worlds.gates import run_gates
from causal_worlds.sample import build_substrate
from causal_worlds.schema import WorldSpec, answer_key
from causal_worlds.worlds import UnknownWorldError

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


@app.command("worlds")
def list_worlds() -> None:
    """List the built-in worlds."""
    for name in worlds.names():
        typer.echo(name)


def _resolve(name: str) -> WorldSpec:
    """Look up a built-in world, exiting cleanly with a message if it is unknown."""
    try:
        return worlds.get(name)
    except UnknownWorldError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None


@app.command()
def grade(world: str, seed: int = 0) -> None:
    """Grade the reference discoverer on a built-in world and print the scores."""
    container = build_container()
    spec = _resolve(world)
    with container.tracer().span("grade"):
        recovered = container.discoverer().recover(build_substrate(spec), seed=seed)
        report = score(recovered, answer_key(spec))
    typer.echo(
        f"directed_shd={report.directed_shd}  f1={report.f1:.2f}  "
        f"confounded_reported={report.confounded_reported}"
    )


@app.command()
def gate(world: str, seed: int = 0) -> None:
    """Run the validity gates on a built-in world and print the admit decision."""
    container = build_container()
    report = run_gates(_resolve(world), discoverer=container.discoverer(), seed=seed)
    typer.echo(f"admitted={report.admitted}  reason={report.reason!r}")
