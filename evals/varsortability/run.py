"""Simulated-DAG leakage controls — varsortability AND R^2-sortability. (Reisach et al.)

For every benchmark world: measure (a) **varsortability** — does marginal *variance* leak the causal
order (0.5 = no; toward 1.0 = yes) — with the `sortnregress` baseline, and (b) **R^2-sortability** —
does *predictability from the rest* leak the order — with the `R2-sortnregress` baseline. Varsortability
is removable by standardization; R^2-sortability is **scale-invariant** and is NOT, so reporting both is
the bar a 2026 causal-discovery reviewer expects ("Beware the Simulated DAG!", + the 2023 follow-up).
Keyless; run from the repo root:

    uv run python evals/varsortability/run.py [benchmark/v0.5]
"""

import json
import sys
from pathlib import Path
from statistics import mean

from causal_worlds.artifact import load_bundle
from causal_worlds.controls import (
    R2SortnregressDiscoverer,
    SortnregressDiscoverer,
    r2sortability,
    varsortability,
)
from causal_worlds.evaluation import score
from causal_worlds.sample import build_substrate
from causal_worlds.schema import answer_key

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "v0.5"
OUT = Path(__file__).parent / BENCH.name
N = 4000
SEED = 7


def main():
    rows = []
    for wdir in sorted(p for p in BENCH.iterdir() if p.is_dir()):
        spec = load_bundle(wdir).spec
        substrate = build_substrate(spec)
        key = answer_key(spec)
        data = substrate.sample(N, seed=SEED).data
        vs = varsortability(data, key.edges, substrate.variables)
        r2s = r2sortability(data, key.edges, substrate.variables)
        sr = score(SortnregressDiscoverer(n=N).recover(substrate, seed=SEED), key)
        r2sr = score(R2SortnregressDiscoverer(n=N).recover(substrate, seed=SEED), key)
        rows.append(
            {
                "world": wdir.name,
                "varsortability": vs,
                "r2sortability": r2s,
                "sortnregress_f1": sr.f1,
                "sortnregress_shd": sr.directed_shd,
                "r2sortnregress_f1": r2sr.f1,
                "r2sortnregress_shd": r2sr.directed_shd,
            }
        )
        print(
            f"{wdir.name}: varsort {vs:.2f} r2sort {r2s:.2f}  "
            f"sortnregress F1 {sr.f1:.2f}  r2-sortnregress F1 {r2sr.f1:.2f}"
        )

    agg = {
        "mean_varsortability": mean(r["varsortability"] for r in rows),
        "mean_r2sortability": mean(r["r2sortability"] for r in rows),
        "mean_sortnregress_f1": mean(r["sortnregress_f1"] for r in rows),
        "mean_sortnregress_shd": mean(r["sortnregress_shd"] for r in rows),
        "mean_r2sortnregress_f1": mean(r["r2sortnregress_f1"] for r in rows),
        "mean_r2sortnregress_shd": mean(r["r2sortnregress_shd"] for r in rows),
    }
    report = {
        "eval": "varsortability",
        "benchmark": BENCH.name,
        "n": N,
        "seed": SEED,
        "per_world": rows,
        "aggregate": agg,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(
        f"\nmean varsortability {agg['mean_varsortability']:.2f} "
        f"(sortnregress F1 {agg['mean_sortnregress_f1']:.2f}) | "
        f"mean R^2-sortability {agg['mean_r2sortability']:.2f} "
        f"(R^2-sortnregress F1 {agg['mean_r2sortnregress_f1']:.2f})"
    )
    print(f"wrote {OUT}/report.json + README.md")


def _write_readme(report):
    a = report["aggregate"]
    var_gamed = a["mean_varsortability"] > 0.65 or a["mean_sortnregress_f1"] > 0.5
    r2_gamed = a["mean_r2sortability"] > 0.65 or a["mean_r2sortnregress_f1"] > 0.5
    verdict = (
        "GAMEABLE — a trivial sorting baseline wins; the worlds leak their causal order."
        if var_gamed or r2_gamed
        else "NOT gamed by either sorting trick — both sortabilities near chance, both baselines weak."
    )
    lines = [
        "# Simulated-DAG leakage controls (varsortability + R^2-sortability)",
        "",
        f"Benchmark `{report['benchmark']}`, n={report['n']}, seed {report['seed']}. Each sortability "
        "is 0.5 when the causal order is NOT readable off that signal and trends to 1.0 when it is "
        "(and the matching trivial baseline would win). Varsortability (marginal variance) is "
        "removable by standardization; **R^2-sortability (predictability) is scale-invariant and is "
        "NOT** — both are reported per the Reisach et al. line of critique.",
        "",
        f"- mean **varsortability {a['mean_varsortability']:.2f}** "
        f"— `sortnregress` mean **F1 {a['mean_sortnregress_f1']:.2f}** "
        f"(SHD {a['mean_sortnregress_shd']:.2f})",
        f"- mean **R^2-sortability {a['mean_r2sortability']:.2f}** "
        f"— `R^2-sortnregress` mean **F1 {a['mean_r2sortnregress_f1']:.2f}** "
        f"(SHD {a['mean_r2sortnregress_shd']:.2f})",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Reproduce: `uv run python evals/varsortability/run.py`.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
