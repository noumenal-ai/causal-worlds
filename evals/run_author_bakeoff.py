"""Author-model bake-off: Sonnet 4.6 vs Opus 4.8, judged by the independent Gemini judge.

Decide the default author model with NUMBERS, not assertion. Each model authors the same prompts,
through the same gates + the same Gemini judge + the same reference grader at the same seed. We pick
the winner on admit-rate, then anti-cliché difficulty, then faithfulness, then attempts.

The report (report.json + README.md) is the shipped, reproducible evidence, versioned with the
package. Run (needs the `llm` extra + both keys in env):

    set -a && . ../.env && set +a && uv run python evals/run_author_bakeoff.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from causal_worlds._version import __version__
from causal_worlds.author import build_claude_author
from causal_worlds.discover import GRADER, GRADER_VERSION, InterventionalCiDiscoverer
from causal_worlds.generate import generate_many
from causal_worlds.judge import DEFAULT_JUDGE_MODEL, build_gemini_judge

SEED = 7
DISCOVERER_N = 6000
MODELS = ["claude-opus-4-8", "claude-sonnet-4-6"]
OUT = Path(__file__).parent / "author-model-bakeoff"

PROMPTS = [
    "A regional coffee chain with weekend demand swings, loyalty pricing, and variable supplier "
    "lead times.",
    "A hospital emergency department: triage staffing, patient inflow, bed availability, and wait "
    "times.",
    "A ride-hailing marketplace with surge pricing, driver supply, rider demand, and cancellations.",
    "A solar microgrid with battery storage, dynamic tariffs, weather swings, and household load.",
    "A contract manufacturing line: machine maintenance, throughput, defect rate, and on-time "
    "delivery.",
    "A SaaS support desk: ticket inflow, staffing, automation coverage, and customer churn.",
    "A last-mile delivery network: courier dispatch, traffic, package volume, and delivery SLA.",
    "A boutique hotel with dynamic room pricing, seasonal events, occupancy, and review scores.",
]


def _world_row(outcome):
    if outcome.world is None:
        return {"prompt": outcome.prompt, "admitted": False, "reason": outcome.reason}
    r = outcome.world.report
    return {
        "prompt": outcome.prompt,
        "admitted": True,
        "attempts": outcome.world.attempts,
        "difficulty": r.difficulty,
        "faithfulness": r.faithfulness,
        "directed_shd": r.grade.directed_shd,
        "f1": r.grade.f1,
    }


def _aggregate(rows):
    admitted = [r for r in rows if r["admitted"]]
    n = len(rows)
    return {
        "n": n,
        "admitted": len(admitted),
        "admit_rate": len(admitted) / n,
        "mean_attempts": mean(r["attempts"] for r in admitted) if admitted else None,
        "mean_difficulty": mean(r["difficulty"] for r in admitted) if admitted else None,
        "mean_faithfulness": mean(r["faithfulness"] for r in admitted) if admitted else None,
        "mean_directed_shd": mean(r["directed_shd"] for r in admitted) if admitted else None,
        "mean_f1": mean(r["f1"] for r in admitted) if admitted else None,
        "worlds": rows,
    }


def _score(agg):
    # rank key: admit-rate, then difficulty, then faithfulness, then fewer attempts (negated).
    return (
        agg["admit_rate"],
        agg["mean_difficulty"] or 0.0,
        agg["mean_faithfulness"] or 0.0,
        -(agg["mean_attempts"] or 99.0),
    )


def main():
    judge = build_gemini_judge(DEFAULT_JUDGE_MODEL)
    discoverer = InterventionalCiDiscoverer(n=DISCOVERER_N)
    results = {}
    for model in MODELS:
        print(f"\n=== author={model} ===")
        author = build_claude_author(model)
        outcomes = generate_many(
            PROMPTS, author=author, judge=judge, discoverer=discoverer, seed=SEED
        )
        rows = [_world_row(o) for o in outcomes]
        agg = _aggregate(rows)
        results[model] = agg
        print(
            f"  admit={agg['admitted']}/{agg['n']}  "
            f"difficulty={agg['mean_difficulty']}  faithfulness={agg['mean_faithfulness']}  "
            f"shd={agg['mean_directed_shd']}  f1={agg['mean_f1']}"
        )

    winner = max(MODELS, key=lambda m: _score(results[m]))
    report = {
        "eval": "author-model-bakeoff",
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "grader": f"{GRADER}@{GRADER_VERSION}",
        "discoverer_n": DISCOVERER_N,
        "prompts": PROMPTS,
        "models": results,
        "winner": winner,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(f"\nWINNER: {winner}\nwrote {OUT}/report.json + README.md")


def _write_readme(report):
    lines = [
        "# Author-model bake-off",
        "",
        f"Decided with numbers, not assertion. Package `v{report['package_version']}`, "
        f"seed {report['seed']}, judge `{report['judge_model']}`, grader `{report['grader']}`.",
        "",
        "| author model | admit | difficulty | faithfulness | directed SHD | F1 | attempts |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, agg in report["models"].items():
        mark = " **(winner)**" if model == report["winner"] else ""

        def fmt(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "-"

        lines.append(
            f"| `{model}`{mark} | {agg['admitted']}/{agg['n']} | {fmt(agg['mean_difficulty'])} | "
            f"{fmt(agg['mean_faithfulness'])} | {fmt(agg['mean_directed_shd'])} | "
            f"{fmt(agg['mean_f1'])} | {fmt(agg['mean_attempts'])} |"
        )
    lines += [
        "",
        f"**Winner: `{report['winner']}`** — ranked on admit-rate, then anti-cliché difficulty, "
        "then faithfulness, then fewer re-author attempts.",
        "",
        "Reproduce: `set -a && . ../.env && set +a && uv run python evals/run_author_bakeoff.py`.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
