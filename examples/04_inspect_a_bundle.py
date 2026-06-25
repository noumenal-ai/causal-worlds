"""Load a shipped benchmark world and read its truth + provenance (no API key).

    uv run python examples/04_inspect_a_bundle.py

Expected output:

    world: benchmark/v0.6/world_01
      prompt          : A hospital emergency department with triage staffing, inflow, beds, and wait times.
      observed columns: ('triage_nurses', 'patient_inflow', 'beds_open', 'wait_time', ...)
      data shape      : (2000, 7)  (rows x observed)
      causal edges    : [('beds_open', 'los_hours'), ('boarding_pressure', 'triage_nurses'), ...]
      confounded pairs: [['beds_open', 'patient_inflow']]  (NOT causal edges)
      difficulty      : name=0.78 structural=2.0
      provenance      : author=claude-opus-4-8 judge=gemini-2.5-flash grader=interventional-ci
      honesty         : Fictional world for benchmarking causal discovery; not a model of any real system.
"""

from pathlib import Path

from causal_worlds import answer_key, load_bundle

BUNDLE = Path("benchmark/v0.6/world_01")


def main() -> None:
    bundle = load_bundle(BUNDLE)
    key = answer_key(bundle.spec)

    print(f"world: {BUNDLE}")
    print(f"  prompt          : {bundle.manifest['prompt']}")
    print(f"  observed columns: {bundle.columns}")
    print(f"  data shape      : {bundle.data.shape}  (rows x observed)")
    print(f"  causal edges    : {sorted(key.edges)}")
    print(f"  confounded pairs: {[sorted(p) for p in key.confounded]}  (NOT causal edges)")
    print(f"  difficulty      : name={bundle.manifest['difficulty']} "
          f"structural={bundle.manifest['structural_difficulty']['score']}")
    print(f"  provenance      : author={bundle.manifest['author_model']} "
          f"judge={bundle.manifest['judge_model']} grader={bundle.manifest['grader']}")
    print(f"  honesty         : {bundle.manifest['honesty']}")


if __name__ == "__main__":
    main()
