"""Live smoke: author a world with Claude, gate it with the independent Gemini judge.

Research/verification script (not shipped, not linted). Needs the `llm` extra and both keys in env:
    set -a && . ../.env && set +a && uv run python spikes/smoke_live.py
"""

from causal_worlds.author import build_claude_author
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.judge import build_gemini_judge
from causal_worlds.generate import NotAdmittedError, generate

PROMPT = (
    "A regional bike-share network: stations, dynamic pricing, weather swings, rebalancing trucks, "
    "and membership churn."
)


def main() -> None:
    author = build_claude_author()  # claude-opus-4-8
    judge = build_gemini_judge()  # gemini-2.5-flash
    discoverer = InterventionalCiDiscoverer(n=4000)
    print(f"author={author._model}  judge={judge._model}")  # noqa: SLF001
    try:
        world = generate(PROMPT, author=author, judge=judge, discoverer=discoverer, seed=7)
    except NotAdmittedError as exc:
        print(f"NOT ADMITTED after {exc.attempts}: {exc.last.reason if exc.last else '?'}")
        return
    r = world.report
    observed = [v.name for v in world.spec.variables if not v.hidden]
    hidden = [v.name for v in world.spec.variables if v.hidden]
    print(f"ADMITTED in {world.attempts} attempt(s)")
    print(f"  observed={observed}")
    print(f"  hidden={hidden}")
    print(f"  difficulty={r.difficulty:.2f}  faithfulness={r.faithfulness:.2f}")
    print(f"  grade: directed_shd={r.grade.directed_shd} f1={r.grade.f1:.2f} "
          f"confounded_reported={r.grade.confounded_reported}")


if __name__ == "__main__":
    main()
