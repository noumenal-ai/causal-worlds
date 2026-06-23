"""Varsortability control — are our worlds gameable by the variance-order trick? (Reisach et al.)

For every benchmark world: measure varsortability (0.5 = the causal order is NOT readable off marginal
variances; toward 1.0 = it is) and run the trivial `sortnregress` baseline. If varsortability is near
0.5 and sortnregress scores poorly, the benchmark is NOT gamed by the variance giveaway — the control
the "Beware the Simulated DAG!" critique demands. Keyless; run from the repo root:

    uv run python evals/varsortability/run.py [benchmark/v0.5]
"""

import json
import sys
from pathlib import Path
from statistics import mean

from causal_worlds.artifact import load_bundle
from causal_worlds.controls import SortnregressDiscoverer, varsortability
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
        sr = score(SortnregressDiscoverer(n=N).recover(substrate, seed=SEED), key)
        rows.append({"world": wdir.name, "varsortability": vs, "sortnregress_f1": sr.f1,
                     "sortnregress_shd": sr.directed_shd})
        print(f"{wdir.name}: varsort {vs:.2f}  sortnregress F1 {sr.f1:.2f} SHD {sr.directed_shd}")

    agg = {
        "mean_varsortability": mean(r["varsortability"] for r in rows),
        "mean_sortnregress_f1": mean(r["sortnregress_f1"] for r in rows),
        "mean_sortnregress_shd": mean(r["sortnregress_shd"] for r in rows),
    }
    report = {"eval": "varsortability", "benchmark": BENCH.name, "n": N, "seed": SEED,
              "per_world": rows, "aggregate": agg}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(f"\nmean varsortability {agg['mean_varsortability']:.2f} | "
          f"sortnregress mean F1 {agg['mean_sortnregress_f1']:.2f}")
    print(f"wrote {OUT}/report.json + README.md")


def _write_readme(report):
    a = report["aggregate"]
    gamed = a["mean_varsortability"] > 0.65 or a["mean_sortnregress_f1"] > 0.5
    verdict = (
        "GAMEABLE — variance leaks the causal order; standardize emitted variances."
        if gamed
        else "NOT gamed by the variance trick — varsortability near chance and sortnregress is weak."
    )
    lines = [
        "# Varsortability control",
        "",
        f"Benchmark `{report['benchmark']}`, n={report['n']}, seed {report['seed']}. Varsortability "
        "(Reisach et al.): 0.5 = the causal order is NOT readable from marginal variances; toward 1.0 "
        "= it is (and the trivial `sortnregress` baseline would win).",
        "",
        f"- mean **varsortability {a['mean_varsortability']:.2f}**",
        f"- `sortnregress` baseline: mean **F1 {a['mean_sortnregress_f1']:.2f}**, "
        f"mean directed SHD {a['mean_sortnregress_shd']:.2f}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Reproduce: `uv run python evals/varsortability/run.py`.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
