"""Live smoke: author a TEMPORAL world with Claude, admit it through the temporal gate (PCMCI+).

Needs the `llm` + `temporal` extras and both keys:
    set -a && . ../.env && set +a && uv run python spikes/smoke_temporal.py
"""

from causal_worlds.author import build_claude_author
from causal_worlds.generate import NotAdmittedError, generate
from causal_worlds.schema import temporal_answer_key

PROMPT = (
    "A reservoir operation over time: upstream rainfall, inflow, controlled release, and downstream "
    "flood risk, with delays as water propagates and storage carries over day to day."
)


def main() -> None:
    author = build_claude_author(complexity="standard", temporal=True)
    print(f"author={author._model} (temporal)")  # noqa: SLF001
    try:
        world = generate(PROMPT, author=author, seed=7)  # temporal gate -> PCMCI+ by default
    except NotAdmittedError as exc:
        print(f"NOT ADMITTED: {exc.last.reason if exc.last else exc}")
        return
    lagged = sorted(temporal_answer_key(world.spec))
    tg = world.report.temporal_grade
    print(f"ADMITTED in {world.attempts} attempt(s)")
    print(f"  observed = {[v.name for v in world.spec.variables if not v.hidden]}")
    print(f"  hidden   = {[v.name for v in world.spec.variables if v.hidden]}")
    print(f"  lagged edges (truth): {lagged}")
    print(f"  PCMCI+ temporal grade: SHD {tg.temporal_shd}  F1 {tg.temporal_f1:.2f}")


if __name__ == "__main__":
    main()
