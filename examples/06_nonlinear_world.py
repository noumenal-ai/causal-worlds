"""A nonlinear world: stopping distance grows with speed**2 — and why that breaks linear discovery.

The built-in `braking` world models auto-emergency-braking physics with an *additive-nonlinear*
mechanism: `braking_distance` follows the kinematic `v**2` law (a `Term(..., transform=SQUARE)`).
On standardized, symmetric speed data the *linear* correlation of speed with braking distance is
~0 — so a linear/PC discoverer that reasons from correlations drops the `speed -> braking_distance`
edge, even though the causal dependence is total. The intervention `do(speed)` makes that dependence
plain. No API key needed.

    uv run python examples/06_nonlinear_world.py

Expected output:

    speed vs braking_distance  Pearson corr = 0.000   <- a linear method sees ~no relationship
    do(speed=2) braking = 3.59   vs   do(speed=0) braking = -0.01   <- but the causal effect is huge

    PC (linear, observational)  : f1=0.6   speed->braking_distance recovered? False
    interventional-CI reference : f1=1.0   speed->braking_distance recovered? True

    The speed->braking_distance edge is in the answer key the whole time; only the method that
    assumes straight lines fails to see it. That is the additive-nonlinear capability (issue #10).
"""

import numpy as np

from causal_worlds import build_substrate, grade_spec, worlds
from causal_worlds.baselines import PcDiscoverer
from causal_worlds.discover import InterventionalCiDiscoverer


def main() -> None:
    spec = worlds.get("braking")  # braking_distance ~ 0.9 * speed**2 - 0.5 * road_grip
    sub = build_substrate(spec, standardize=False)  # raw units, so do() effects are visible
    bd = sub.variables.index("braking_distance")
    sp = sub.variables.index("speed")

    sample = sub.sample(60_000, seed=0)
    corr = float(np.corrcoef(sample.data[:, sp], sample.data[:, bd])[0, 1])
    hi = float(sub.sample(60_000, seed=1, do={"speed": 2.0}).data[:, bd].mean())
    lo = float(sub.sample(60_000, seed=1, do={"speed": 0.0}).data[:, bd].mean())
    print(
        f"speed vs braking_distance  Pearson corr = {corr:.3f}   <- a linear method sees ~no relationship"
    )
    print(
        f"do(speed=2) braking = {hi:.2f}   vs   do(speed=0) braking = {lo:.2f}   <- but the causal effect is huge"
    )
    print()

    edge = ("speed", "braking_distance")
    pc_edges = PcDiscoverer().recover(build_substrate(spec), seed=0)
    pc = grade_spec(spec, PcDiscoverer())
    ref_edges = InterventionalCiDiscoverer().recover(build_substrate(spec), seed=0)
    ref = grade_spec(spec, InterventionalCiDiscoverer())
    print(
        f"PC (linear, observational)  : f1={pc.f1}   speed->braking_distance recovered? {edge in pc_edges}"
    )
    print(
        f"interventional-CI reference : f1={ref.f1}   speed->braking_distance recovered? {edge in ref_edges}"
    )


if __name__ == "__main__":
    main()
