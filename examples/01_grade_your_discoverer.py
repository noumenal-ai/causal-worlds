"""Implement a Discoverer and score it against a world's declared answer key (no API key).

    uv run python examples/01_grade_your_discoverer.py
"""

import numpy as np

from causal_worlds import InterventionalCiDiscoverer, grade_spec, worlds


class CorrelationDiscoverer:
    """A deliberately naive baseline: connect strongly-correlated observed variables.

    A Discoverer just needs `recover(substrate, *, seed) -> set[(src, dst)]`. The substrate is the
    executable world: `substrate.sample(n, seed=...)` for observational data, or pass
    `do={"name": value}` to intervene.
    """

    def recover(self, substrate, *, seed):
        sample = substrate.sample(4000, seed=seed)
        names = substrate.variables
        corr = np.corrcoef(sample.data, rowvar=False)
        return {
            (names[i], names[j])
            for i in range(len(names))
            for j in range(len(names))
            if i < j and abs(corr[i, j]) > 0.5
        }


def main() -> None:
    spec = worlds.get("coffee")  # a hidden confounder + a regime sign-flip

    naive = grade_spec(spec, CorrelationDiscoverer())
    reference = grade_spec(spec, InterventionalCiDiscoverer())

    print("naive correlation :", naive)
    print("interventional-ci :", reference)
    print()
    print(f"correlation keeps {naive.confounded_reported} spurious confounded edge(s) as causal;")
    print(f"the interventional grader keeps {reference.confounded_reported} — that gap is the point.")


if __name__ == "__main__":
    main()
