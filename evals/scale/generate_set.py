"""Generate the scaled benchmark set (v0.5) — 36 worlds across easy/standard/hard complexity.

v0.4 found the difficulty-predicts-error question is underpowered at n=12 with a narrow range. This
deliberately spreads structural difficulty (the author's complexity knob: easy = no traps, standard =
one confounder + one regime flip, hard = two of each) over a varied prompt pool, so the re-run
analyses have range and power. Needs the `llm` extra + both keys:

    set -a && . ../.env && set +a && uv run python evals/scale/generate_set.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from causal_worlds.artifact import Provenance, save_bundle
from causal_worlds.author import build_claude_author
from causal_worlds.difficulty import structural_difficulty
from causal_worlds.discover import GRADER, GRADER_VERSION, InterventionalCiDiscoverer
from causal_worlds.generate import NotAdmittedError, generate
from causal_worlds.judge import DEFAULT_JUDGE_MODEL, build_gemini_judge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmark" / "v0.5"
SEED = 7
N = 4000
LEVELS = ["easy", "standard", "hard"]

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
    "A film-render farm: job scheduling, node utilization, render time, and reruns.",
    "A telecom network operation: capacity provisioning, traffic, congestion, and dropped calls.",
    "A restaurant kitchen line: prep staffing, ticket inflow, service time, and waste.",
    "A vaccination campaign: clinic staffing, appointment demand, throughput, and no-shows.",
    "A fashion retailer markdown operation: discount depth, traffic, sell-through, and margin.",
    "A municipal recycling operation: collection routing, contamination, throughput, and cost.",
]


def main():
    judge = build_gemini_judge(DEFAULT_JUDGE_MODEL)
    discoverer = InterventionalCiDiscoverer(n=N)
    authors = {level: build_claude_author(complexity=level) for level in LEVELS}
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for i, prompt in enumerate(PROMPTS):
        level = LEVELS[i % len(LEVELS)]
        try:
            world = generate(prompt, author=authors[level], judge=judge, discoverer=discoverer, seed=SEED)
        except NotAdmittedError as exc:
            index.append({"admitted": False, "complexity": level, "prompt": prompt, "reason": str(exc)})
            print(f"world_{i:02d} [{level}] NOT ADMITTED: {exc}")
            continue
        slug = f"world_{i:02d}"
        provenance = Provenance(
            author_model="claude-opus-4-8",
            judge_model=DEFAULT_JUDGE_MODEL,
            grader=GRADER,
            grader_version=GRADER_VERSION,
            seed=SEED,
            n_rows=2000,
            complexity=level,
            created_at=datetime.now(UTC).isoformat(),
        )
        save_bundle(world, OUT / slug, provenance=provenance)
        sd = structural_difficulty(world.spec)
        r = world.report
        index.append({
            "slug": slug,
            "admitted": True,
            "complexity": level,
            "prompt": prompt,
            "attempts": world.attempts,
            "difficulty": r.difficulty,
            "faithfulness": r.faithfulness,
            "structural_score": sd.score,
            "directed_shd": r.grade.directed_shd,
            "f1": r.grade.f1,
        })
        print(f"{slug} [{level}] diff {r.difficulty:.2f} struct {sd.score} shd {r.grade.directed_shd}")

    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    admitted = sum(1 for e in index if e["admitted"])
    print(f"\n{admitted}/{len(index)} admitted -> {OUT}")


if __name__ == "__main__":
    main()
