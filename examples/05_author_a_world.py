"""Author a world from a sentence with Claude, gated by the independent Gemini judge.

Needs the `llm` extra and both keys in the environment:

    uv add 'causal-worlds[llm]'
    set -a && . ../.env && set +a
    uv run python examples/05_author_a_world.py

Expected output (illustrative — depends on the live models, so a real run varies):

    admitted in 2 attempt(s)
      observed : ['surge_multiplier', 'driver_supply', 'rider_demand', 'cancellations', 'churn']
      hidden   : ['L_market_heat']
      difficulty 0.62  faithfulness 0.90
      reference grader: directed_shd=1 f1=0.86
"""

from causal_worlds import NotAdmittedError, generate
from causal_worlds.author import build_claude_author
from causal_worlds.judge import build_gemini_judge

PROMPT = "a ride-hailing marketplace with surge pricing, driver supply, and rider churn"


def main() -> None:
    author = build_claude_author(complexity="hard")  # easy | standard | hard
    judge = build_gemini_judge()  # a different model family than the author

    try:
        world = generate(PROMPT, author=author, judge=judge, seed=7)
    except NotAdmittedError as exc:
        print(f"not admitted after {exc.attempts} attempt(s): {exc.last and exc.last.reason}")
        return

    report = world.report
    observed = [v.name for v in world.spec.variables if not v.hidden]
    hidden = [v.name for v in world.spec.variables if v.hidden]
    print(f"admitted in {world.attempts} attempt(s)")
    print(f"  observed : {observed}")
    print(f"  hidden   : {hidden}")
    print(f"  difficulty {report.difficulty:.2f}  faithfulness {report.faithfulness:.2f}")
    print(f"  reference grader: directed_shd={report.grade.directed_shd} f1={report.grade.f1:.2f}")


if __name__ == "__main__":
    main()
