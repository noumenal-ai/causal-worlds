"""Author a world from a sentence with Claude, gated by the independent Gemini judge.

Needs the `llm` extra and both keys in the environment:

    uv add 'causal-worlds[llm]'
    set -a && . ../.env && set +a
    uv run python examples/05_author_a_world.py

Two modes (see the ``anti_cliche`` flag / the CLI ``--playground`` flag):

* **benchmark mode** (default): a world is rejected if it is guessable from its variable names/roles —
  the published benchmark must not be name-guessable. Everyday operations often fail this on purpose.
* **playground mode** (``anti_cliche=False``): keep the faithfulness check and the difficulty score,
  but never reject on guessability — the "describe a world and just get it" path.

This example tries benchmark mode first and **falls back to playground** if the world is too
guessable, so you always end up with a world. Expected output (a real playground run of the grid
prompt; live models vary):

    benchmark mode: not admitted (T4 cliché: names+roles recover it (prior F1 0.57 >= 0.5))
    -> retrying in playground mode (anti-cliché advisory)...
    admitted in 1 attempt(s)  [playground]
      observed : ['tou_price', 'solar_gen', 'battery_charge', 'peak_regime', 'home_demand', 'grid_import']
      hidden   : ['weather']
      difficulty 0.41 (advisory)  faithfulness 0.90
      reference grader: directed_shd=1 f1=0.92
"""

from causal_worlds import NotAdmittedError, generate
from causal_worlds.author import build_claude_author
from causal_worlds.judge import build_gemini_judge

PROMPT = "a regional power grid with rooftop solar, home batteries, and time-of-use pricing"


def main() -> None:
    author = build_claude_author(complexity="adversarial")  # easy | standard | hard | adversarial
    judge = build_gemini_judge()  # a different model family than the author

    playground = False
    try:
        world = generate(PROMPT, author=author, judge=judge, seed=1)
    except NotAdmittedError as exc:
        print(f"benchmark mode: not admitted ({exc.last and exc.last.reason})")
        if not (exc.last and "cliché" in exc.last.reason):
            return
        print("-> retrying in playground mode (anti-cliché advisory)...")
        playground = True
        world = generate(PROMPT, author=author, judge=judge, seed=1, anti_cliche=False)

    report = world.report
    observed = [v.name for v in world.spec.variables if not v.hidden]
    hidden = [v.name for v in world.spec.variables if v.hidden]
    advisory = " (advisory)" if playground else ""
    print(f"admitted in {world.attempts} attempt(s)  [{'playground' if playground else 'benchmark'}]")
    print(f"  observed : {observed}")
    print(f"  hidden   : {hidden}")
    print(f"  difficulty {report.difficulty:.2f}{advisory}  faithfulness {report.faithfulness:.2f}")
    print(f"  reference grader: directed_shd={report.grade.directed_shd} f1={report.grade.f1:.2f}")


if __name__ == "__main__":
    main()
