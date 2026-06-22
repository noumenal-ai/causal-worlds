"""Structural difficulty vs error — does structure predict the collapse name-guessability missed?

v0.3 found that name-guessability difficulty does NOT predict how badly the standard methods fail.
This re-analysis (no new discovery runs — it reuses the crossover report) asks whether the *structural*
difficulty (confounded pairs + regime sign-flips) does. The compelling, non-tautological signal: the
**interventional advantage** (reference F1 minus mean observational F1) should rise with structural
difficulty — interventions help most exactly where the structure is hard.

Run from the repo root (keyless): uv run python evals/structural-difficulty/run_analysis.py
"""

import json
import sys
from pathlib import Path
from statistics import correlation, mean

from causal_worlds.artifact import load_bundle
from causal_worlds.difficulty import structural_difficulty

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "v0.2"
_XOVER_DIR = ROOT / "evals" / "baseline-crossover"
# v0.2's crossover report lives at the top level; later sets are nested by name.
CROSSOVER = next(
    p for p in (_XOVER_DIR / BENCH.name / "report.json", _XOVER_DIR / "report.json") if p.exists()
)
OUT = Path(__file__).parent / BENCH.name
OBSERVATIONAL = ["pc", "fci", "gies"]  # ges errored in the crossover (numpy-2)


def _corr(xs, ys):
    ok = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
    xs2, ys2 = [x for x, _ in ok], [y for _, y in ok]
    if len(xs2) < 2 or len(set(xs2)) < 2 or len(set(ys2)) < 2:
        return None
    return correlation(xs2, ys2)


def main():
    crossover = json.loads(CROSSOVER.read_text())["per_world"]
    rows = []
    for wdir in sorted(p for p in BENCH.iterdir() if p.is_dir()):
        bundle = load_bundle(wdir)
        sd = structural_difficulty(bundle.spec)
        cell = crossover[wdir.name]["methods"]
        obs_skel = [cell[m]["skeleton_shd_mean"] for m in OBSERVATIONAL if "skeleton_shd_mean" in cell[m]]
        obs_f1 = [cell[m]["directed_f1_mean"] for m in OBSERVATIONAL if "directed_f1_mean" in cell[m]]
        interv = cell["interventional-ci"]
        rows.append({
            "world": wdir.name,
            "name_difficulty": bundle.manifest["difficulty"],
            "structural_score": sd.score,
            "confounded_pairs": sd.confounded_pairs,
            "sign_flips": sd.sign_flips,
            "obs_skeleton_shd": mean(obs_skel) if obs_skel else None,
            "obs_f1": mean(obs_f1) if obs_f1 else None,
            "interv_f1": interv.get("directed_f1_mean"),
            "interv_advantage": (interv["directed_f1_mean"] - mean(obs_f1)) if obs_f1 else None,
        })

    struct = [r["structural_score"] for r in rows]
    name = [r["name_difficulty"] for r in rows]
    correlations = {
        "structural_vs_obs_skeleton_shd": _corr(struct, [r["obs_skeleton_shd"] for r in rows]),
        "structural_vs_interventional_advantage": _corr(struct, [r["interv_advantage"] for r in rows]),
        "name_vs_obs_skeleton_shd": _corr(name, [r["obs_skeleton_shd"] for r in rows]),
        "name_vs_interventional_advantage": _corr(name, [r["interv_advantage"] for r in rows]),
    }
    report = {
        "eval": "structural-difficulty",
        "benchmark": "v0.2",
        "source_crossover": "evals/baseline-crossover/report.json",
        "rows": rows,
        "correlations": correlations,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    for k, v in correlations.items():
        print(f"{k}: {v}")
    print(f"wrote {OUT}/report.json + README.md")


def _write_readme(report):
    c = report["correlations"]

    def f(x):
        return f"{x:+.2f}" if isinstance(x, (int, float)) else "—"

    lines = [
        "# Structural difficulty vs error",
        "",
        "v0.3 found name-guessability difficulty doesn't predict discovery error. This re-analysis "
        "(reusing the crossover report — no new runs) tests whether *structural* difficulty "
        "(confounded pairs + regime sign-flips) does.",
        "",
        "| correlation | name-guessability | structural |",
        "|---|---|---|",
        f"| vs observational skeleton-SHD | {f(c['name_vs_obs_skeleton_shd'])} | "
        f"{f(c['structural_vs_obs_skeleton_shd'])} |",
        f"| vs interventional advantage (ΔF1) | {f(c['name_vs_interventional_advantage'])} | "
        f"{f(c['structural_vs_interventional_advantage'])} |",
        "",
        "*Interventional advantage* = reference F1 − mean observational F1 per world: how much the "
        "`do()`-based grader gains over the observational toolbox.",
        "",
        "Reproduce (keyless): `uv run python evals/structural-difficulty/run_analysis.py`.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
