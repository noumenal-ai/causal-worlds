"""Temporal crossover — do the standard TS methods recover the lagged structure of `supply`, and do
the latent-naive ones fall for the hidden confounder where latent-aware LPCMCI does not?

`supply` hides L (logistics) confounding leadtime~cost contemporaneously, on top of autoregressive
lead time + inventory. We grade each TS baseline against the lagged answer-key and check whether it
puts a spurious *causal* edge on the confounded pair. Needs the `temporal` extra; no API key.

    uv run python evals/temporal-crossover/run.py
"""

import json
from pathlib import Path

from causal_worlds import temporal_answer_key, temporal_score, worlds
from causal_worlds.sample import build_substrate
from causal_worlds.temporal_baselines import TEMPORAL_BASELINES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
WORLD = "supply"
N = 2000
SEED = 7
CONFOUNDED = [("leadtime", "cost", 0), ("cost", "leadtime", 0)]  # the lag-0 trap in `supply`


def main():
    spec = worlds.get(WORLD)
    substrate = build_substrate(spec)
    truth = temporal_answer_key(spec)
    rows = {}
    for name, cls in TEMPORAL_BASELINES.items():
        print(f"running {name} ...")
        try:
            recovered = cls(n=N).recover_temporal(substrate, seed=SEED)
        except Exception as exc:  # noqa: BLE001 - record the failure honestly, keep going
            rows[name] = {"errored": f"{type(exc).__name__}: {exc}"[:80]}
            continue
        report = temporal_score(recovered, truth)
        rows[name] = {
            "temporal_shd": report.temporal_shd,
            "temporal_f1": round(report.temporal_f1, 3),
            "n_recovered": report.n_recovered,
            "confounded_kept_as_causal": any(e in recovered for e in CONFOUNDED),
        }
        print(f"  {name}: {rows[name]}")

    report = {
        "eval": "temporal-crossover",
        "world": WORLD,
        "n": N,
        "seed": SEED,
        "n_truth_lagged_edges": len(truth),
        "latent_naive": ["pcmci+", "varlingam", "granger"],
        "latent_aware": ["lpcmci"],
        "methods": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(f"\nwrote {OUT}/report.json + README.md")


def _write_readme(report):
    rows = report["methods"]

    def cell(name, key, fmt="{}"):
        v = rows[name].get(key)
        return rows[name].get("errored", "—") if "errored" in rows[name] else fmt.format(v)

    lines = [
        "# Temporal crossover",
        "",
        f"Time-series discovery on the temporal built-in `{report['world']}` (n={report['n']}, seed "
        f"{report['seed']}, {report['n_truth_lagged_edges']} true lagged edges). `supply` hides a "
        "logistics confounder L driving lead time ~ cost, on top of autoregressive lead time + "
        "inventory. Latent-naive methods (PCMCI+, VARLiNGAM, Granger) assume no hidden confounders; "
        "LPCMCI is latent-aware.",
        "",
        "| method | temporal SHD ↓ | temporal F1 ↑ | recovered | kept confounded pair as causal? |",
        "|---|---|---|---|---|",
    ]
    for name in TEMPORAL_BASELINES:
        lines.append(
            f"| `{name}` | {cell(name, 'temporal_shd')} | {cell(name, 'temporal_f1')} | "
            f"{cell(name, 'n_recovered')} | {cell(name, 'confounded_kept_as_causal')} |"
        )
    lines += [
        "",
        "Reproduce: `uv run python evals/temporal-crossover/run.py` (needs the `temporal` extra).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
