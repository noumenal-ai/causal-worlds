"""Load a shipped benchmark world and read its truth + provenance (no API key).

    uv run python examples/02_inspect_a_bundle.py
"""

from pathlib import Path

from causal_worlds import answer_key, load_bundle

BUNDLE = Path("benchmark/v0.5/world_01")


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
