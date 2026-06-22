"""The causal-worlds command-line interface (the construction-from-use edge)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from causal_worlds import __version__, worlds
from causal_worlds.artifact import Provenance, save_bundle
from causal_worlds.container import Container, build_container
from causal_worlds.evaluation import score
from causal_worlds.gates import run_gates
from causal_worlds.generate import AdmittedWorld, NotAdmittedError, generate_many
from causal_worlds.generate import generate as generate_world
from causal_worlds.protocols import Author, Judge
from causal_worlds.sample import build_substrate
from causal_worlds.schema import WorldSpec, answer_key
from causal_worlds.worlds import UnknownWorldError

_LLM_HINT = "Live generation needs the 'llm' extra and API keys: uv add 'causal-worlds[llm]'"

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


def _live(container: Container) -> tuple[Author, Judge]:
    """Build the live author + judge, exiting with a hint if the ``llm`` extra is missing."""
    try:
        return container.author(), container.judge()
    except ImportError:
        typer.echo(_LLM_HINT, err=True)
        raise typer.Exit(code=1) from None


def _provenance(container: Container, seed: int) -> Provenance:
    """Assemble manifest provenance, reading the wall clock here at the shell edge."""
    grader, version = container.grader_provenance()
    return Provenance(
        author_model=container.settings.author_model,
        judge_model=container.settings.judge_model,
        grader=grader,
        grader_version=version,
        seed=seed,
        n_rows=container.settings.bundle_rows,
        created_at=datetime.now(UTC).isoformat(),
    )


def _admit_summary(world: AdmittedWorld) -> dict[str, object]:
    """A compact, JSON-friendly row describing an admitted world (for the benchmark index)."""
    report = world.report
    grade = report.grade
    return {
        "admitted": True,
        "prompt": world.prompt,
        "attempts": world.attempts,
        "difficulty": report.difficulty,
        "faithfulness": report.faithfulness,
        "directed_shd": grade.directed_shd if grade else None,
        "f1": grade.f1 if grade else None,
    }


@app.command()
def generate(prompt: str, out: Path, seed: int = 0) -> None:
    """Author a world from PROMPT, gate it, and write the admitted bundle to OUT."""
    container = build_container()
    author, judge = _live(container)
    try:
        world = generate_world(
            prompt,
            author=author,
            judge=judge,
            discoverer=container.discoverer(),
            seed=seed,
            max_attempts=container.settings.max_attempts,
        )
    except NotAdmittedError as exc:
        typer.echo(f"not admitted: {exc}", err=True)
        raise typer.Exit(code=1) from None
    save_bundle(world, out, provenance=_provenance(container, seed))
    typer.echo(f"admitted -> {out}  difficulty={world.report.difficulty:.2f}")


@app.command()
def benchmark(prompts_file: Path, out: Path, seed: int = 0) -> None:
    """Generate a benchmark set — one admitted world per prompt line in PROMPTS_FILE, under OUT."""
    container = build_container()
    author, judge = _live(container)
    prompts = [
        line.strip()
        for line in prompts_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    outcomes = generate_many(
        prompts,
        author=author,
        judge=judge,
        discoverer=container.discoverer(),
        seed=seed,
        max_attempts=container.settings.max_attempts,
    )

    out.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = []
    for i, outcome in enumerate(outcomes):
        if outcome.world is None:
            index.append({"admitted": False, "prompt": outcome.prompt, "reason": outcome.reason})
            continue
        slug = f"world_{i:02d}"
        save_bundle(outcome.world, out / slug, provenance=_provenance(container, seed))
        index.append({"slug": slug, **_admit_summary(outcome.world)})

    (out / "index.json").write_text(json.dumps(index, indent=2))
    admitted = sum(1 for row in index if row["admitted"])
    typer.echo(f"{admitted}/{len(index)} admitted -> {out}")
