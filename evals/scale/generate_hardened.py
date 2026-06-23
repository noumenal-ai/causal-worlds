"""Generate the hardened benchmark set (v0.6) — adversarial author + the strict anti-cliché gate.

v0.5 was name-guessable (name-only prior F1 0.71 vs 0.20 chance; #12). This regenerates a small set
with the **adversarial** author tier (the obvious name-based guess is wrong) under the v0.19 strict T4
gate (admit only at difficulty >= 0.5, plus a name+role-blind control near chance). Capped small (<=30
worlds) for the private eval at that scale; scale up later as needed. Needs the `llm` extra + keys:

    set -a && . ./.env && set +a
    uv run --extra llm --extra discover python evals/scale/generate_hardened.py [count]
"""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from causal_worlds.artifact import Provenance, save_bundle
from causal_worlds.author import build_claude_author
from causal_worlds.difficulty import structural_difficulty
from causal_worlds.discover import GRADER, GRADER_VERSION, InterventionalCiDiscoverer
from causal_worlds.generate import NotAdmittedError, generate
from causal_worlds.judge import DEFAULT_JUDGE_MODEL, build_gemini_judge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "v0.6"
SEED = 7
N = 4000
MAX_ATTEMPTS = 4  # bound re-author cost per world under the strict gate
DEFAULT_COUNT = 30  # cap (the user allows up to 30)
TRANSIENT_RETRIES = 5  # retry a transient provider error (e.g. Gemini 503) before giving up

PROMPTS = [
    "A regional coffee chain with weekend demand swings and variable supplier lead times.",
    "A hospital emergency department with triage staffing, inflow, beds, and wait times.",
    "A ride-hailing marketplace with surge pricing, driver supply, demand, and cancellations.",
    "A solar microgrid with battery storage, dynamic tariffs, weather, and household load.",
    "A contract manufacturing line: maintenance, throughput, defect rate, and on-time delivery.",
    "A SaaS support desk: ticket inflow, staffing, automation coverage, and churn.",
    "A last-mile delivery network: courier dispatch, traffic, package volume, and SLA.",
    "A boutique hotel with dynamic room pricing, seasonal events, occupancy, and reviews.",
    "A grocery cold chain: refrigeration setpoints, energy cost, spoilage, and stockouts.",
    "A call-center workforce plan: scheduling, call volume, handle time, and abandonment.",
    "A bike-share network: rebalancing, dynamic pricing, weather, docks, and churn.",
    "A wastewater treatment plant: aeration control, inflow load, energy use, and effluent quality.",
    "An outpatient clinic with appointment scheduling, no-shows, room utilization, and overtime.",
    "A pharmacy inventory operation: reorder points, demand, stockouts, and expiry write-offs.",
    "An airport ground-handling operation: gate assignment, turnaround, delays, and fuel use.",
    "An EV-charging network: pricing, session demand, queueing, and grid-draw penalties.",
    "A transit bus line: frequency, ridership, bunching, and on-time performance.",
    "A district heating system: supply temperature, demand, pump energy, and complaints.",
    "A wind farm: maintenance scheduling, downtime, output, and curtailment.",
    "A data-center cooling operation: setpoints, IT load, PUE, and thermal alarms.",
    "A semiconductor fab: tool maintenance, wafer starts, yield, and cycle time.",
    "A craft brewery: batch scheduling, fermentation temperature, throughput, and spoilage.",
    "A 3D-print farm: queue policy, printer uptime, throughput, and reject rate.",
    "A textile dyeing line: batch size, water temperature, color defects, and rework.",
    "A content-moderation queue: reviewer staffing, inflow, backlog, and error rate.",
    "A fraud-review pipeline: alert thresholds, analyst load, false positives, and losses.",
    "A vertical farm: lighting schedule, nutrient dosing, growth rate, and energy cost.",
    "A dairy herd operation: feed mix, milking frequency, yield, and health incidents.",
    "A warehouse pick-pack operation: slotting, labor, throughput, and mispicks.",
    "A port container yard: crane scheduling, dwell time, congestion, and demurrage.",
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    temporal = "--temporal" in sys.argv  # author lagged worlds (DSA's native time-series substrate)
    count = int(args[0]) if args else DEFAULT_COUNT
    out = Path(args[1]) if len(args) > 1 else OUT
    prompts = PROMPTS[:count]
    judge = build_gemini_judge(DEFAULT_JUDGE_MODEL)
    discoverer = InterventionalCiDiscoverer(n=N)
    author = build_claude_author(complexity="adversarial", temporal=temporal)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, prompt in enumerate(prompts):
        slug = f"world_{i:02d}"
        if (out / slug).exists():  # resume: don't re-spend on worlds already written
            man = json.loads((out / slug / "manifest.json").read_text())
            index.append({"slug": slug, "admitted": True, "prompt": prompt,
                          "difficulty": man.get("difficulty"), "resumed": True})
            print(f"{slug} resumed (exists)")
            continue
        try:
            world = _generate_resilient(prompt, author, judge, discoverer)
        except NotAdmittedError as exc:
            index.append({"admitted": False, "prompt": prompt, "reason": str(exc)[:90]})
            print(f"{slug} NOT ADMITTED ({world_attempts(exc)}): {str(exc)[:70]}")
            continue
        provenance = Provenance(
            author_model="claude-opus-4-8",
            judge_model=DEFAULT_JUDGE_MODEL,
            grader=GRADER,
            grader_version=GRADER_VERSION,
            seed=SEED,
            n_rows=2000,
            complexity="adversarial",
            created_at=datetime.now(UTC).isoformat(),
        )
        save_bundle(world, out / slug, provenance=provenance)
        sd = structural_difficulty(world.spec)
        r = world.report
        temporal_f1 = r.temporal_grade.temporal_f1 if r.temporal_grade else None
        index.append({
            "slug": slug,
            "admitted": True,
            "prompt": prompt,
            "attempts": world.attempts,
            "difficulty": r.difficulty,  # None for temporal worlds (T4 anti-cliché is skipped)
            "faithfulness": r.faithfulness,
            "structural_score": sd.score,
            "directed_shd": r.grade.directed_shd if r.grade else None,
            "f1": r.grade.f1 if r.grade else None,
            "temporal_f1": temporal_f1,
        })
        diff_str = f"diff {r.difficulty:.2f}" if r.difficulty is not None else f"tF1 {temporal_f1:.2f}"
        print(f"{slug} {diff_str} struct {sd.score} attempts {world.attempts}")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    admitted = [e for e in index if e["admitted"]]
    scores = [e["difficulty"] if e["difficulty"] is not None else e.get("temporal_f1") for e in admitted]
    scores = [s for s in scores if s is not None]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n{len(admitted)}/{len(index)} admitted -> {out} | mean difficulty/tF1 {mean_score:.2f}")


def _generate_resilient(prompt, author, judge, discoverer):
    """Run generate(), retrying transient provider errors (e.g. Gemini 503); admit/reject pass through."""
    for attempt in range(TRANSIENT_RETRIES):
        try:
            return generate(
                prompt, author=author, judge=judge, discoverer=discoverer,
                seed=SEED, max_attempts=MAX_ATTEMPTS,
            )
        except NotAdmittedError:
            raise  # a real verdict, not a transient failure
        except Exception as exc:  # noqa: BLE001 - retry any transient API/transport error
            if attempt == TRANSIENT_RETRIES - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  transient {type(exc).__name__}, retry {attempt + 1}/{TRANSIENT_RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def world_attempts(exc):
    """Best-effort attempt count from a NotAdmittedError (for the log line)."""
    return getattr(exc, "attempts", "?")


if __name__ == "__main__":
    main()
