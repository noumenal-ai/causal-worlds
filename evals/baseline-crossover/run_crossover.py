"""Baseline crossover — the decisive experiment, made information-fair (#9).

Two questions, separated honestly:

1. **Observational track** — do the standard observational methods (PC, GES, FCI) recover the
   structure, and do they keep the hidden-confounded pair as a *causal* edge (the trap)?
2. **Interventional track (information-fair)** — give the *same interventional budget* (pooled
   observational + per-variable do() environments) to the causal-sufficiency methods too (`pc+do`,
   `fci+do`, plus GIES which is natively interventional) and the reference latent-aware grader. If
   the causal-sufficiency methods *still* keep the confounded pair with interventions in hand, the
   dividing line is **latent-awareness**, not data access — an identifiability result, not "we beat
   the toolbox."

Reported: per-method skeleton-SHD / directed-F1 / confounded-kept; the **interventional advantage**
ΔF1 = F1(reference) − F1(method) with a **bootstrap CI** (the primary, non-tautological signal); and
difficulty-vs-error correlations with bootstrap CIs.

Needs the `discover` extra; no API key. Run from the repo root:

    uv run python evals/baseline-crossover/run_crossover.py [benchmark/v0.5]
"""

import json
import sys
from pathlib import Path
from statistics import correlation, mean, pstdev

import numpy as np

from causal_worlds.artifact import load_bundle
from causal_worlds.baselines import (
    DagmaDiscoverer,
    DirectLingamDiscoverer,
    FciDiscoverer,
    GesDiscoverer,
    GiesDiscoverer,
    PcDiscoverer,
)
from causal_worlds.discover import GRADER, GRADER_VERSION, InterventionalCiDiscoverer
from causal_worlds.evaluation import score
from causal_worlds.sample import build_substrate
from causal_worlds.schema import answer_key

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "v0.2"
OUT = Path(__file__).parent / BENCH.name
N = 4000
SEEDS = [7, 11, 23]
BOOT_REPS = 2000
REFERENCE = "interventional-ci"

# (method name) -> (kind, class, data-access label). "kind" routes how it is run.
METHODS: dict[str, tuple[str, type | None, str]] = {
    REFERENCE: ("reference", None, "interventional"),
    "pc": ("clearn", PcDiscoverer, "observational"),
    "ges": ("clearn", GesDiscoverer, "observational"),
    "fci": ("clearn", FciDiscoverer, "observational"),
    "dagma": ("adjacency", DagmaDiscoverer, "observational"),
    "directlingam": ("adjacency", DirectLingamDiscoverer, "observational"),
    "gies": ("gies", GiesDiscoverer, "interventional"),
    "pc+do": ("clearn-do", PcDiscoverer, "interventional"),
    "fci+do": ("clearn-do", FciDiscoverer, "interventional"),
}


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


def _run_baseline(kind, cls, substrate, key, seed):
    if kind == "clearn-do":
        result = cls(n=N, interventional=True).detail(substrate, seed=seed)
    else:
        result = cls(n=N).detail(substrate, seed=seed)
    report = score(result.edges, key)
    return {
        "skeleton_shd": len(result.skeleton ^ _truth_skeleton(key)),
        "directed_f1": report.f1,
        "confounded_kept": _confounded_kept_baseline(result, key),
    }


def _score_world(method, substrate, key):
    """Run one method across seeds; return per-metric mean/std, or an error record."""
    kind, cls, _ = METHODS[method]
    runs = []
    for seed in SEEDS:
        try:
            runs.append(
                _run_reference(substrate, key, seed)
                if kind == "reference"
                else _run_baseline(kind, cls, substrate, key, seed)
            )
        except Exception as exc:  # noqa: BLE001 - record the failure honestly, keep going
            return {"errored": f"{type(exc).__name__}: {exc}"[:80]}
    return {
        "skeleton_shd_mean": mean(r["skeleton_shd"] for r in runs),
        "skeleton_shd_std": pstdev(r["skeleton_shd"] for r in runs),
        "directed_f1_mean": mean(r["directed_f1"] for r in runs),
        "confounded_kept_mean": mean(r["confounded_kept"] for r in runs),
    }


def _bootstrap_ci(values, statistic, reps, seed, level=0.95):
    """Percentile bootstrap CI of ``statistic`` over ``values`` (a list of per-world numbers/tuples)."""
    if len(values) < 2:  # noqa: PLR2004
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    stats = []
    for _ in range(reps):
        sample = arr[rng.integers(0, n, n)]
        value = statistic(sample)
        if value is not None:
            stats.append(value)
    if not stats:
        return None
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def _delta_f1(per_world, method):
    """Per-world interventional advantage: F1(reference) − F1(method), where both scored."""
    out = []
    for w in per_world.values():
        ref, cell = w["methods"][REFERENCE], w["methods"][method]
        if "directed_f1_mean" in ref and "directed_f1_mean" in cell:
            out.append(ref["directed_f1_mean"] - cell["directed_f1_mean"])
    return out


def main():
    worlds = sorted(p for p in BENCH.iterdir() if p.is_dir())
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
            "methods": {m: _score_world(m, substrate, key) for m in METHODS},
        }
        print(
            f"{wdir.name} (diff {difficulty:.2f}): "
            + "  ".join(
                f"{m}={per_world[wdir.name]['methods'][m].get('skeleton_shd_mean', 'ERR')}"
                for m in METHODS
            )
        )

    report = {
        "eval": "baseline-crossover",
        "benchmark": BENCH.name,
        "grader": f"{GRADER}@{GRADER_VERSION}",
        "n": N,
        "seeds": SEEDS,
        "methods": list(METHODS),
        "per_world": per_world,
        "aggregate": _aggregate(per_world),
        "interventional_advantage": _interventional_advantage(per_world),
        "difficulty_vs_error": _difficulty_vs_error(per_world),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(f"\nwrote {OUT}/report.json + README.md")


def _aggregate(per_world):
    agg = {}
    for m in METHODS:
        rows = [
            w["methods"][m] for w in per_world.values() if "skeleton_shd_mean" in w["methods"][m]
        ]
        errored = sum(1 for w in per_world.values() if "errored" in w["methods"][m])
        agg[m] = {
            "data": METHODS[m][2],
            "worlds_scored": len(rows),
            "errored": errored,
            "mean_skeleton_shd": mean(r["skeleton_shd_mean"] for r in rows) if rows else None,
            "mean_directed_f1": mean(r["directed_f1_mean"] for r in rows) if rows else None,
            "total_confounded_kept": sum(r["confounded_kept_mean"] for r in rows) if rows else None,
        }
    return agg


def _interventional_advantage(per_world):
    """ΔF1 = F1(reference) − F1(method), mean + bootstrap CI — the primary, non-tautological signal."""
    out = {}
    for m in METHODS:
        if m == REFERENCE:
            continue
        deltas = _delta_f1(per_world, m)
        if not deltas:
            continue
        out[m] = {
            "mean_delta_f1": mean(deltas),
            "ci95": _bootstrap_ci(deltas, lambda s: float(s.mean()), BOOT_REPS, seed=1),
            "n": len(deltas),
        }
    return out


def _difficulty_vs_error(per_world):
    """Pearson corr of difficulty with skeleton-SHD error, per method, with a bootstrap CI."""
    out = {}
    for m in METHODS:
        pairs = [
            (w["difficulty"], w["methods"][m]["skeleton_shd_mean"])
            for w in per_world.values()
            if "skeleton_shd_mean" in w["methods"][m]
        ]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        usable = len(pairs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1  # noqa: PLR2004
        out[m] = {
            "pearson": correlation(xs, ys) if usable else None,
            "ci95": _bootstrap_ci(pairs, _corr_of, BOOT_REPS, seed=2) if usable else None,
        }
    return out


def _corr_of(sample):
    """Pearson correlation of a bootstrap resample of (x, y) rows, or None if degenerate."""
    xs, ys = sample[:, 0].tolist(), sample[:, 1].tolist()
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return None
    return correlation(xs, ys)


def _write_readme(report):
    agg = report["aggregate"]
    adv = report["interventional_advantage"]
    corr = report["difficulty_vs_error"]

    def f(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

    def ci(c):
        return f"[{c[0]:.2f}, {c[1]:.2f}]" if c else "—"

    lines = [
        "# Baseline crossover — information-fair (#9)",
        "",
        f"Every world in benchmark `{report['benchmark']}` vs the standard discoverers and the "
        f"reference grader `{report['grader']}`, at n={report['n']}, averaged over seeds "
        f"{report['seeds']}. The **interventional track** (`+do`, gies, reference) gives the "
        "causal-sufficiency methods the *same interventional budget* (pooled observational + "
        "per-variable do() environments) as the latent-aware reference. Skeleton-SHD = adjacency "
        "errors; confounded-kept = hidden-confounded pairs reported as *causal* (the trap; lower "
        "better).",
        "",
        "| method | data | worlds | mean skeleton-SHD | mean directed F1 | confounded-kept |",
        "|---|---|---|---|---|---|",
    ]
    for m in report["methods"]:
        a = agg[m]
        err = f" (+{a['errored']} errored)" if a["errored"] else ""
        lines.append(
            f"| `{m}` | {a['data']} | {a['worlds_scored']}{err} | {f(a['mean_skeleton_shd'])} | "
            f"{f(a['mean_directed_f1'])} | {f(a['total_confounded_kept'])} |"
        )
    lines += [
        "",
        "### Interventional advantage — ΔF1 = F1(reference) − F1(method), 95% bootstrap CI",
        "",
        "| method | mean ΔF1 | 95% CI |",
        "|---|---|---|",
    ]
    for m, a in adv.items():
        lines.append(f"| `{m}` | {f(a['mean_delta_f1'])} | {ci(a['ci95'])} |")
    lines += [
        "",
        "### Difficulty vs skeleton-SHD error — Pearson r, 95% bootstrap CI",
        "",
        "| method | r | 95% CI |",
        "|---|---|---|",
    ]
    for m in report["methods"]:
        c = corr[m]
        lines.append(f"| `{m}` | {f(c['pearson'])} | {ci(c['ci95'])} |")
    lines += [
        "",
        "**Read:** the headline is the **interventional track** — if `pc+do`, `fci+do`, and `gies` "
        "still keep confounded pairs and post a positive ΔF1 (CI excluding 0) while the latent-aware "
        "`interventional-ci` reaches confounded-kept 0, the dividing line is **latent-awareness, not "
        "interventions** (an identifiability result). Difficulty-vs-error is descriptive (n is small, "
        "predictors discrete); the bootstrap CI shows how wide it really is.",
        "",
        "Reproduce: `uv run python evals/baseline-crossover/run_crossover.py benchmark/v0.5` "
        "(needs the `discover` extra).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
