"""Baseline crossover — the decisive experiment (v0.3, go/no-go).

For every world in the shipped benchmark set, run the standard discoverers (PC, GES, FCI, GIES) and
the reference interventional-CI grader; score skeleton recovery, directed F1, and — the trap — whether
each method reports the hidden-confounded pair as a *causal* edge. Then correlate per-world difficulty
with error: the finding is that observational/score-based error rises with difficulty while the
interventional grader holds.

Needs the `discover` extra (causal-learn + gies); no API key. Run from the repo root:

    uv run python evals/baseline-crossover/run_crossover.py
"""

import json
import sys
from pathlib import Path
from statistics import correlation, mean, pstdev

from causal_worlds.artifact import load_bundle
from causal_worlds.baselines import FciDiscoverer, GesDiscoverer, GiesDiscoverer, PcDiscoverer
from causal_worlds.discover import GRADER, GRADER_VERSION, InterventionalCiDiscoverer
from causal_worlds.evaluation import score
from causal_worlds.sample import build_substrate
from causal_worlds.schema import answer_key

ROOT = Path(__file__).resolve().parents[2]
# Benchmark dir from argv[1] (default v0.2); outputs go to OUT/<benchmark name>.
BENCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "v0.2"
OUT = Path(__file__).parent / BENCH.name
N = 4000
SEEDS = [7, 11, 23]
BASELINES = {
    "pc": PcDiscoverer,
    "ges": GesDiscoverer,
    "fci": FciDiscoverer,
    "gies": GiesDiscoverer,
}
REFERENCE = "interventional-ci"


def _truth_skeleton(key):
    return {frozenset(edge) for edge in key.edges}


def _confounded_kept_baseline(result, key):
    """Confounded pairs the baseline reports as a directed causal edge (the trap; bidirected is OK)."""
    kept = 0
    for pair in key.confounded:
        directed = any((a, b) in result.edges for a, b in (tuple(pair), tuple(pair)[::-1]))
        if directed and pair not in result.bidirected:
            kept += 1
    return kept


def _run_reference(substrate, key, seed):
    edges = InterventionalCiDiscoverer(n=N).recover(substrate, seed=seed)
    report = score(edges, key)
    skeleton = {frozenset(e) for e in edges}
    return {
        "skeleton_shd": len(skeleton ^ _truth_skeleton(key)),
        "directed_f1": report.f1,
        "confounded_kept": report.confounded_reported,
    }


def _run_baseline(cls, substrate, key, seed):
    result = cls(n=N).detail(substrate, seed=seed)
    report = score(result.edges, key)
    return {
        "skeleton_shd": len(result.skeleton ^ _truth_skeleton(key)),
        "directed_f1": report.f1,
        "confounded_kept": _confounded_kept_baseline(result, key),
    }


def _score_world(method, substrate, key):
    """Run one method across seeds; return per-metric mean/std, or an error record."""
    runs = []
    for seed in SEEDS:
        try:
            runs.append(
                _run_reference(substrate, key, seed)
                if method == REFERENCE
                else _run_baseline(BASELINES[method], substrate, key, seed)
            )
        except Exception as exc:  # noqa: BLE001 - record the failure honestly, keep going
            return {"errored": f"{type(exc).__name__}: {exc}"[:80]}
    return {
        "skeleton_shd_mean": mean(r["skeleton_shd"] for r in runs),
        "skeleton_shd_std": pstdev(r["skeleton_shd"] for r in runs),
        "directed_f1_mean": mean(r["directed_f1"] for r in runs),
        "confounded_kept_mean": mean(r["confounded_kept"] for r in runs),
    }


def main():
    worlds = sorted(p for p in BENCH.iterdir() if p.is_dir())
    methods = [REFERENCE, *BASELINES]
    per_world = {}
    for wdir in worlds:
        bundle = load_bundle(wdir)
        spec = bundle.spec
        substrate = build_substrate(spec)
        key = answer_key(spec)
        difficulty = bundle.manifest["difficulty"]
        per_world[wdir.name] = {
            "difficulty": difficulty,
            "n_confounded": len(key.confounded),
            "methods": {m: _score_world(m, substrate, key) for m in methods},
        }
        print(f"{wdir.name} (diff {difficulty:.2f}): " + "  ".join(
            f"{m}={per_world[wdir.name]['methods'][m].get('skeleton_shd_mean', 'ERR')}"
            for m in methods
        ))

    report = {
        "eval": "baseline-crossover",
        "benchmark": "v0.2",
        "grader": f"{GRADER}@{GRADER_VERSION}",
        "n": N,
        "seeds": SEEDS,
        "methods": methods,
        "per_world": per_world,
        "aggregate": _aggregate(per_world, methods),
        "difficulty_vs_error": _difficulty_vs_error(per_world, methods),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(f"\nwrote {OUT}/report.json + README.md")


def _aggregate(per_world, methods):
    agg = {}
    for m in methods:
        rows = [w["methods"][m] for w in per_world.values() if "skeleton_shd_mean" in w["methods"][m]]
        errored = sum(1 for w in per_world.values() if "errored" in w["methods"][m])
        agg[m] = {
            "worlds_scored": len(rows),
            "errored": errored,
            "mean_skeleton_shd": mean(r["skeleton_shd_mean"] for r in rows) if rows else None,
            "mean_directed_f1": mean(r["directed_f1_mean"] for r in rows) if rows else None,
            "total_confounded_kept": sum(r["confounded_kept_mean"] for r in rows) if rows else None,
        }
    return agg


def _difficulty_vs_error(per_world, methods):
    """Pearson correlation of per-world difficulty with skeleton-SHD error, per method."""
    out = {}
    for m in methods:
        xs, ys = [], []
        for w in per_world.values():
            cell = w["methods"][m]
            if "skeleton_shd_mean" in cell:
                xs.append(w["difficulty"])
                ys.append(cell["skeleton_shd_mean"])
        out[m] = (
            correlation(xs, ys)
            if len(xs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1
            else None
        )
    return out


def _write_readme(report):
    agg = report["aggregate"]
    corr = report["difficulty_vs_error"]

    def f(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

    lines = [
        "# Baseline crossover — the decisive experiment",
        "",
        f"Every world in benchmark `{report['benchmark']}` vs the standard discoverers and the "
        f"reference grader `{report['grader']}`, at n={report['n']}, averaged over seeds "
        f"{report['seeds']}. Skeleton-SHD = adjacency errors (lower better); confounded-kept = "
        "hidden-confounded pairs reported as *causal* (the trap; lower better).",
        "",
        "| method | worlds | mean skeleton-SHD | mean directed F1 | confounded-kept | "
        "corr(difficulty, error) |",
        "|---|---|---|---|---|---|",
    ]
    for m in report["methods"]:
        a = agg[m]
        err = f" (+{a['errored']} errored)" if a["errored"] else ""
        lines.append(
            f"| `{m}` | {a['worlds_scored']}{err} | {f(a['mean_skeleton_shd'])} | "
            f"{f(a['mean_directed_f1'])} | {f(a['total_confounded_kept'])} | {f(corr[m])} |"
        )
    lines += [
        "",
        "**Read:** if the observational/score-based methods (pc/ges/fci/gies) keep confounded pairs "
        "and post higher skeleton-SHD while `interventional-ci` stays near zero, the benchmark is "
        "non-trivial. If `corr(difficulty, error)` is positive for the standard methods and ~flat "
        "for the grader, difficulty is a real instrument (error rises with how guessable a world is).",
        "",
        "Reproduce: `uv run python evals/baseline-crossover/run_crossover.py` (needs the `discover` "
        "extra).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
