"""The causal-worlds command-line interface (the construction-from-use edge)."""

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from causal_worlds import __version__, worlds
from causal_worlds.artifact import Provenance, save_bundle
from causal_worlds.bench import grade_bundle
from causal_worlds.brief import is_complete, render
from causal_worlds.container import Container, build_container
from causal_worlds.elicit import Session, force_ready, respond, start_session
from causal_worlds.evaluation import score
from causal_worlds.gates import run_gates
from causal_worlds.generate import AdmittedWorld, NotAdmittedError, generate_many
from causal_worlds.generate import generate as generate_world
from causal_worlds.protocols import Author, Discoverer, Elicitor, Judge
from causal_worlds.sample import build_substrate
from causal_worlds.schema import WorldSpec, answer_key
from causal_worlds.worlds import UnknownWorldError

_GO_WORDS = {"go", "ready", "done", "generate"}  # user can force handoff with any of these
_MAX_TURNS = 12  # bound the clarify loop

_LLM_HINT = "Live generation needs the 'llm' extra and API keys: uv add 'causal-worlds[llm]'"

app = typer.Typer(
    help="Generate and grade fictional causal worlds.",
    no_args_is_help=True,
)


def _load_dotenv() -> None:
    """Load a local .env into the environment (so provider + Langfuse keys are picked up).

    Best practice for Langfuse: env vars must be loaded before its client is constructed. Graceful
    if python-dotenv isn't installed.
    """
    try:
        from dotenv import load_dotenv  # noqa: PLC0415 - optional convenience
    except ImportError:
        return
    load_dotenv()


@app.callback()
def main() -> None:
    """Generate and grade fictional causal worlds."""
    _load_dotenv()


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


def _spec_of(world: str) -> WorldSpec:
    """Resolve ``world`` as a persisted bundle dir if it exists, else as a built-in world name."""
    path = Path(world)
    if (path / "spec.json").exists():
        from causal_worlds.artifact import load_bundle  # noqa: PLC0415 - keep IO out of import time

        return load_bundle(path).spec
    return _resolve(world)


@app.command()
def viz(world: str, fmt: str = typer.Option("mermaid", "--format", help="mermaid | dot")) -> None:
    """Print a world's SCM as a Mermaid (default) or Graphviz-DOT graph (hidden confounders shown).

    ``world`` is a built-in name (e.g. ``coffee``) or a persisted bundle directory.
    """
    from causal_worlds.viz import to_dot, to_mermaid  # noqa: PLC0415

    spec = _spec_of(world)
    renderers = {"mermaid": to_mermaid, "dot": to_dot}
    if fmt not in renderers:
        typer.echo("--format must be 'mermaid' or 'dot'", err=True)
        raise typer.Exit(code=1)
    typer.echo(renderers[fmt](spec))


def _load_discoverer(path: str) -> Discoverer:
    """Import a discoverer from a ``module:Class`` path and instantiate it (no args)."""
    module_name, _, attr = path.partition(":")
    if not attr:
        typer.echo("--discoverer must be 'module:Class'", err=True)
        raise typer.Exit(code=1)
    discoverer: Discoverer = getattr(importlib.import_module(module_name), attr)()
    return discoverer


@app.command("score")
def score_world(bundle: Path, discoverer: str = "", seed: int = 0) -> None:
    """Grade a discoverer on a persisted world BUNDLE (default: the reference interventional-CI)."""
    grader = _load_discoverer(discoverer) if discoverer else None
    report = grade_bundle(bundle, grader, seed=seed)
    typer.echo(
        f"directed_shd={report.directed_shd}  skeleton_shd={report.skeleton_shd}  "
        f"f1={report.f1:.2f}  confounded_reported={report.confounded_reported}"
    )


def _live(container: Container) -> tuple[Author, Judge]:
    """Build the live author + judge, exiting with a hint if the ``llm`` extra is missing."""
    try:
        return container.author(), container.judge()
    except ImportError:
        typer.echo(_LLM_HINT, err=True)
        raise typer.Exit(code=1) from None


def _provenance(container: Container, seed: int, *, anti_cliche: bool = True) -> Provenance:
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
        anti_cliche=anti_cliche,
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


def _admit_detail(world: AdmittedWorld) -> str:
    """A short human line about an admitted world (cross-sectional difficulty or temporal F1)."""
    report = world.report
    if report.difficulty is not None:
        return f"difficulty={report.difficulty:.2f}"
    if report.temporal_grade is not None:
        return f"temporal F1={report.temporal_grade.temporal_f1:.2f}"
    return "admitted"


def _author_and_save(  # noqa: PLR0913 - a shell glue function over the generate pipeline's inputs
    container: Container,
    author: Author,
    judge: Judge,
    prompt: str,
    out: Path,
    seed: int,
    *,
    anti_cliche: bool = True,
) -> None:
    """Author a world from ``prompt``, gate it, save the bundle (the shared generate path)."""
    tracer = container.tracer()
    try:
        world = generate_world(
            prompt,
            author=author,
            judge=judge,
            discoverer=container.discoverer(),
            seed=seed,
            max_attempts=container.settings.max_attempts,
            tracer=tracer,
            anti_cliche=anti_cliche,
        )
    except NotAdmittedError as exc:
        typer.echo(f"not admitted: {exc}", err=True)
        if anti_cliche and exc.last is not None and "cliché" in exc.last.reason:
            typer.echo(
                "hint: that is the benchmark anti-cliché gate (the world was guessable from its "
                "names/roles). Re-run with --playground to author it anyway — guessability then "
                "becomes an advisory difficulty score instead of a rejection.",
                err=True,
            )
        raise typer.Exit(code=1) from None
    finally:
        tracer.flush()
    save_bundle(world, out, provenance=_provenance(container, seed, anti_cliche=anti_cliche))
    typer.echo(f"admitted -> {out}  {_admit_detail(world)}")


@app.command()
def generate(prompt: str, out: Path, seed: int = 0, *, playground: bool = False) -> None:
    """Author a world from PROMPT, gate it, and write the admitted bundle to OUT.

    By default a world is rejected if it is guessable from its variable names/roles (benchmark
    mode). Pass --playground to keep the faithfulness check and difficulty score but never reject
    on guessability — the "describe a world and get it" path.
    """
    container = build_container()
    author, judge = _live(container)
    _author_and_save(container, author, judge, prompt, out, seed, anti_cliche=not playground)


def _live_elicitor(container: Container) -> Elicitor:
    """Build the live elicitor, exiting with a hint if the ``llm`` extra is missing."""
    try:
        return container.elicitor()
    except ImportError:
        typer.echo(_LLM_HINT, err=True)
        raise typer.Exit(code=1) from None


def _run_dialogue(elicitor: Elicitor) -> Session:
    """Run the clarify loop until the brief is ready, the user says 'go', or turns run out."""
    session = start_session()
    typer.echo(f"\n{session.question}")
    for _ in range(_MAX_TURNS):
        answer = typer.prompt("you").strip()
        session = (
            force_ready(session)
            if answer.lower() in _GO_WORDS
            else respond(elicitor, session, answer)
        )
        typer.echo(f"\n--- brief so far ---\n{render(session.brief)}\n")
        if session.ready:
            return session
        typer.echo(str(session.question))
    typer.echo("(turn limit reached — generating from the brief so far)", err=True)
    return force_ready(session)


@app.command()
def elicit(out: Path, seed: int = 0, *, playground: bool = False) -> None:
    """Interactively elicit a world brief through dialogue, then author + gate it into OUT.

    Pass --playground to skip the benchmark anti-cliché rejection (see ``generate``).
    """
    container = build_container()
    elicitor = _live_elicitor(container)
    author, judge = _live(container)
    session = _run_dialogue(elicitor)
    if not is_complete(session.brief):
        typer.echo("note: the brief is still thin; authoring from what we have.", err=True)
    _author_and_save(
        container, author, judge, render(session.brief), out, seed, anti_cliche=not playground
    )


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
    tracer = container.tracer()
    try:
        outcomes = generate_many(
            prompts,
            author=author,
            judge=judge,
            discoverer=container.discoverer(),
            seed=seed,
            max_attempts=container.settings.max_attempts,
            tracer=tracer,
        )
    finally:
        tracer.flush()

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
