"""Name-only baseline — does the benchmark survive name anonymization? (#9, Caliper-style)

The anti-cliché claim is that our worlds test *discovery*, not memorized priors. The operational
proof: a name-only LLM baseline (guess the graph from variable names + roles, no data) must score
near **chance** — and, decisively, **anonymizing the names** (relabel to X1..Xn) must not lower it
further (there was no name signal to lose).

For every world we score three things against the truth (directed F1):
- **named**   — the judge's prior from the real names + roles,
- **anon**    — the judge's prior after anonymization (X1..Xn),
- **null**    — a random same-size graph (the chance floor, keyless).

Anti-cliché holds if `named` and `anon` are both close to `null`. The named−anon gap is how much the
names leak. Needs the `llm` extra + a Gemini key for named/anon; `null` is keyless. From the repo root:

    uv run python evals/name-only-baseline/run.py [benchmark/v0.5]
"""

import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np

_RETRIES = 5  # transient Gemini 503s: retry with backoff before giving up on a world


def _with_retries(call, *, what):
    """Run a flaky LLM call, retrying transient failures with linear backoff."""
    for attempt in range(_RETRIES):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - retry any transient provider error
            if attempt == _RETRIES - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  {what}: {type(exc).__name__}, retry {attempt + 1}/{_RETRIES} in {wait}s")
            time.sleep(wait)
    return None  # unreachable

from causal_worlds.anonymize import anonymize_spec
from causal_worlds.artifact import load_bundle
from causal_worlds.evaluation import f1
from causal_worlds.judge import DEFAULT_JUDGE_MODEL, build_gemini_judge
from causal_worlds.schema import answer_key

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmark" / "v0.5"
OUT = Path(__file__).parent / BENCH.name
NULL_REPS = 2000
BOOT_REPS = 2000


def _random_null_f1(truth, names, rng, reps):
    """Mean directed F1 of random same-size graphs vs the truth — the chance floor."""
    pairs = [(a, b) for a in names for b in names if a != b]
    k = len(truth)
    if k == 0 or len(pairs) < k:
        return 0.0
    total = 0.0
    for _ in range(reps):
        chosen = rng.choice(len(pairs), size=k, replace=False)
        total += f1(frozenset(pairs[int(i)] for i in chosen), truth)
    return total / reps


def _bootstrap_ci(values, reps, seed, level=0.95):
    if len(values) < 2:  # noqa: PLR2004
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(reps)]
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def main():
    judge = build_gemini_judge()
    rng = np.random.default_rng(7)
    rows = []
    for wdir in sorted(p for p in BENCH.iterdir() if p.is_dir()):
        spec = load_bundle(wdir).spec
        truth = answer_key(spec).edges
        names = tuple(v.name for v in spec.variables if not v.hidden)
        anon_spec, _ = anonymize_spec(spec)
        anon_truth = answer_key(anon_spec).edges

        try:
            named_f1 = f1(_with_retries(lambda: judge.prior_edges(spec), what="named"), truth)
            anon_f1 = f1(
                _with_retries(lambda: judge.prior_edges(anon_spec), what="anon"), anon_truth
            )
            blind_f1 = f1(
                _with_retries(lambda: judge.prior_edges(spec, blind=True), what="blind"), truth
            )
        except Exception as exc:  # noqa: BLE001 - skip a persistently-failing world, keep the rest
            print(f"{wdir.name}: SKIPPED ({type(exc).__name__})")
            continue
        null_f1 = _random_null_f1(truth, names, rng, NULL_REPS)
        rows.append({
            "world": wdir.name, "named_f1": named_f1, "anon_f1": anon_f1,
            "blind_f1": blind_f1, "null_f1": null_f1,
        })
        print(
            f"{wdir.name}: named {named_f1:.2f}  name-blind {anon_f1:.2f}  "
            f"name+role-blind {blind_f1:.2f}  null {null_f1:.2f}"
        )

    if not rows:
        print("no worlds scored (provider unavailable) — nothing written")
        return
    agg = {
        "scored": len(rows),
        "mean_named_f1": mean(r["named_f1"] for r in rows),
        "mean_anon_f1": mean(r["anon_f1"] for r in rows),
        "mean_blind_f1": mean(r["blind_f1"] for r in rows),
        "mean_null_f1": mean(r["null_f1"] for r in rows),
        "named_ci95": _bootstrap_ci([r["named_f1"] for r in rows], BOOT_REPS, seed=1),
        "anon_ci95": _bootstrap_ci([r["anon_f1"] for r in rows], BOOT_REPS, seed=2),
        "blind_ci95": _bootstrap_ci([r["blind_f1"] for r in rows], BOOT_REPS, seed=3),
        "null_ci95": _bootstrap_ci([r["null_f1"] for r in rows], BOOT_REPS, seed=4),
    }
    report = {
        "eval": "name-only-baseline",
        "benchmark": BENCH.name,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "per_world": rows,
        "aggregate": agg,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(
        f"\nmean named F1 {agg['mean_named_f1']:.2f} | name-blind {agg['mean_anon_f1']:.2f} | "
        f"name+role-blind {agg['mean_blind_f1']:.2f} | null {agg['mean_null_f1']:.2f}"
    )


def _write_readme(report):
    a = report["aggregate"]

    def ci(c):
        return f"[{c[0]:.2f}, {c[1]:.2f}]" if c else "—"

    # Anti-cliché holds if the named guess is no better than chance by more than a small margin.
    near_chance = a["mean_named_f1"] <= a["mean_null_f1"] + 0.15  # noqa: PLR2004
    verdict = (
        "ANTI-CLICHÉ HOLDS — the name-only baseline is near chance; the benchmark tests discovery, "
        "not memorized priors."
        if near_chance
        else "LEAKY — priors beat chance; worlds are guessable from names and/or roles."
    )
    lines = [
        "# Name-only baseline (anti-cliché certificate)",
        "",
        f"Benchmark `{report['benchmark']}`, judge `{report['judge_model']}`. Directed F1 of a "
        "prior-only LLM guess (no data) vs the truth, against the random-graph chance floor, at "
        "three disclosure levels (Caliper-style). If a level beats `null`, the answer leaks at that "
        "level: **named** = names + roles; **name-blind** = names anonymized to `X1..Xn`, roles "
        "kept; **name+role-blind** = names anonymized AND roles hidden (should sit at chance).",
        "",
        f"- **named** F1 {a['mean_named_f1']:.2f}  {ci(a['named_ci95'])}",
        f"- **name-blind** F1 {a['mean_anon_f1']:.2f}  {ci(a['anon_ci95'])}",
        f"- **name+role-blind** F1 {a['mean_blind_f1']:.2f}  {ci(a['blind_ci95'])}",
        f"- **null** (chance) F1 {a['mean_null_f1']:.2f}  {ci(a['null_ci95'])}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Reproduce: `uv run python evals/name-only-baseline/run.py benchmark/v0.5` "
        "(needs the `llm` extra + a Gemini key).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
