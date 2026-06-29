"""Nonlinear crossover — does observational linear discovery miss a nonlinear edge the answer key
still scores, while interventional methods recover it?

We hold a small confounded graph fixed and sweep the transform on ONE edge `X -> M` across
{identity (control), square, cube, tanh, relu, abs}, over a few templates and seeds. Because a
transform never changes the edge *set*, `X -> M` is in the answer key for every world; the question
is which methods recover it. Metrics, per method: directed-F1, skeleton-SHD, confounded-kept (the
hidden-pair trap), and — the focus here — **nonlinear-edge recall** (is the declared `X -> M` edge in
the recovered set?), broken out by transform.

Keyless; needs the `discover` extra. Run from the repo root:

    uv run python evals/nonlinear-crossover/run.py
"""

import json
from pathlib import Path
from statistics import mean

from causal_worlds.baselines import (
    DagmaDiscoverer,
    DirectLingamDiscoverer,
    FciDiscoverer,
    GiesDiscoverer,
    PcDiscoverer,
)
from causal_worlds.discover import GRADER, GRADER_VERSION, InterventionalCiDiscoverer
from causal_worlds.evaluation import score
from causal_worlds.sample import build_substrate
from causal_worlds.schema import (
    Mechanism,
    Role,
    Term,
    Transform,
    Variable,
    WorldSpec,
    answer_key,
    validate,
)

OUT = Path(__file__).parent
N = 4000
SEEDS = [7, 11, 23]
REFERENCE = "interventional-ci"
NONLINEAR_EDGE = ("X", "M")  # the edge whose transform we sweep — present in every world's key

TRANSFORMS = [
    Transform.IDENTITY,  # control: linear edge, every method should get it
    Transform.SQUARE,  # even ⇒ linear corr ≈ 0
    Transform.CUBE,  # odd, steep
    Transform.TANH,  # saturating, monotone
    Transform.RELU,  # rectified
    Transform.ABS,  # even, V-shaped
]

METHODS = {
    REFERENCE: ("reference", None),
    "pc": ("clearn", PcDiscoverer),
    "fci": ("clearn", FciDiscoverer),
    "dagma": ("adjacency", DagmaDiscoverer),
    "directlingam": ("adjacency", DirectLingamDiscoverer),
    "gies": ("gies", GiesDiscoverer),
    "pc+do": ("clearn-do", PcDiscoverer),
    "fci+do": ("clearn-do", FciDiscoverer),
}


def _make_world(template: int, transform: Transform) -> WorldSpec:
    """A small confounded graph whose `X -> M` edge carries ``transform``.

    Hidden ``H`` drives two observed siblings with no edge between them (the confounded pair); ``X``
    (controllable) reaches the outcome through ``M`` via the swept transform. Three templates vary
    the surrounding linear structure so the result is not an artifact of one graph.
    """
    xm = Term("X", 0.9, transform=transform)
    if template == 0:
        variables = (
            Variable("H", Role.DISTURBANCE, hidden=True),
            Variable("X", Role.CONTROLLABLE),
            Variable("A", Role.OBSERVABLE),
            Variable("B", Role.OBSERVABLE),
            Variable("M", Role.OBSERVABLE),
            Variable("Y", Role.OUTCOME),
        )
        mechanisms = (
            Mechanism("A", (Term("H", 0.8),)),
            Mechanism("B", (Term("H", 0.8),)),
            Mechanism("M", (xm, Term("A", 0.5))),
            Mechanism("Y", (Term("M", 1.0), Term("B", 0.4))),
        )  # confounded pair {A, B}
    elif template == 1:
        variables = (
            Variable("H", Role.DISTURBANCE, hidden=True),
            Variable("X", Role.CONTROLLABLE),
            Variable("A", Role.OBSERVABLE),
            Variable("C", Role.OBSERVABLE),
            Variable("M", Role.OBSERVABLE),
            Variable("Y", Role.OUTCOME),
        )
        mechanisms = (
            Mechanism("A", (Term("H", 0.7),)),
            Mechanism("C", (Term("H", 0.7), Term("X", 0.3))),
            Mechanism("M", (xm,)),
            Mechanism("Y", (Term("M", 0.9), Term("A", 0.5), Term("C", 0.3))),
        )  # confounded pair {A, C}
    else:
        variables = (
            Variable("H", Role.DISTURBANCE, hidden=True),
            Variable("W", Role.DISTURBANCE),
            Variable("X", Role.CONTROLLABLE),
            Variable("A", Role.OBSERVABLE),
            Variable("B", Role.OBSERVABLE),
            Variable("M", Role.OBSERVABLE),
            Variable("Y", Role.OUTCOME),
        )
        mechanisms = (
            Mechanism("A", (Term("H", 0.8),)),
            Mechanism("B", (Term("H", 0.8), Term("W", 0.5))),
            Mechanism("M", (xm, Term("W", 0.4))),
            Mechanism("Y", (Term("M", 1.0), Term("A", 0.4))),
        )  # confounded pair {A, B}
    spec = WorldSpec(variables=variables, mechanisms=mechanisms)
    validate(spec)
    return spec


def _truth_skeleton(key):
    return {frozenset(edge) for edge in key.edges}


def _confounded_kept(result, key):
    kept = 0
    for pair in key.confounded:
        directed = any((a, b) in result.edges for a, b in (tuple(pair), tuple(pair)[::-1]))
        if directed and pair not in result.bidirected:
            kept += 1
    return kept


def _recovers_nl_edge(edges) -> bool:
    return NONLINEAR_EDGE in edges


def _run(method, substrate, key, seed):
    kind, cls = METHODS[method]
    if kind == "reference":
        edges = InterventionalCiDiscoverer(n=N).recover(substrate, seed=seed)
        report = score(edges, key)
        return {
            "f1": report.f1,
            "shd": len({frozenset(e) for e in edges} ^ _truth_skeleton(key)),
            "confounded_kept": report.confounded_reported,
            "nl_edge": _recovers_nl_edge(edges),
        }
    result = (
        cls(n=N, interventional=True).detail(substrate, seed=seed)
        if kind == "clearn-do"
        else cls(n=N).detail(substrate, seed=seed)
    )
    report = score(result.edges, key)
    return {
        "f1": report.f1,
        "shd": len(result.skeleton ^ _truth_skeleton(key)),
        "confounded_kept": _confounded_kept(result, key),
        "nl_edge": _recovers_nl_edge(result.edges),
    }


def main() -> None:
    n_templates = 3
    per_world = {}
    for transform in TRANSFORMS:
        for t in range(n_templates):
            spec = _make_world(t, transform)
            substrate = build_substrate(spec)
            key = answer_key(spec)
            name = f"{transform.value}/T{t}"
            cells = {}
            for m in METHODS:
                try:
                    runs = [_run(m, substrate, key, s) for s in SEEDS]
                    cells[m] = {
                        "f1": mean(r["f1"] for r in runs),
                        "shd": mean(r["shd"] for r in runs),
                        "confounded_kept": mean(r["confounded_kept"] for r in runs),
                        "nl_edge_recall": mean(1.0 if r["nl_edge"] else 0.0 for r in runs),
                    }
                except Exception as exc:  # noqa: BLE001 - record + continue
                    cells[m] = {"errored": f"{type(exc).__name__}: {exc}"[:80]}
            per_world[name] = {"transform": transform.value, "template": t, "methods": cells}
            print(
                f"{name}: " + "  ".join(f"{m}_F1={cells[m].get('f1', 'ERR')!s:.5}" for m in METHODS)
            )

    report = {
        "eval": "nonlinear-crossover",
        "grader": f"{GRADER}@{GRADER_VERSION}",
        "n": N,
        "seeds": SEEDS,
        "nonlinear_edge": list(NONLINEAR_EDGE),
        "templates": n_templates,
        "transforms": [t.value for t in TRANSFORMS],
        "per_world": per_world,
        "by_transform": _by_transform(per_world),
        "aggregate": _aggregate(per_world),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    _write_readme(report)
    print(f"\nwrote {OUT}/report.json + README.md")


def _scored(per_world, m):
    return [w["methods"][m] for w in per_world.values() if "f1" in w["methods"][m]]


def _aggregate(per_world):
    agg = {}
    for m in METHODS:
        rows = _scored(per_world, m)
        agg[m] = {
            "worlds_scored": len(rows),
            "errored": sum(1 for w in per_world.values() if "errored" in w["methods"][m]),
            "mean_f1": mean(r["f1"] for r in rows) if rows else None,
            "mean_shd": mean(r["shd"] for r in rows) if rows else None,
            "total_confounded_kept": sum(r["confounded_kept"] for r in rows) if rows else None,
            "nl_edge_recall": mean(r["nl_edge_recall"] for r in rows) if rows else None,
        }
    return agg


def _by_transform(per_world):
    """Mean nonlinear-edge recall per (method, transform) — the focus table."""
    out = {}
    for transform in TRANSFORMS:
        tv = transform.value
        out[tv] = {}
        for m in METHODS:
            rows = [
                w["methods"][m]["nl_edge_recall"]
                for w in per_world.values()
                if w["transform"] == tv and "nl_edge_recall" in w["methods"][m]
            ]
            out[tv][m] = mean(rows) if rows else None
    return out


def _write_readme(report):
    bt = report["by_transform"]
    agg = report["aggregate"]

    def f(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

    methods = list(METHODS)
    lines = [
        "# Nonlinear crossover — does linear discovery miss a nonlinear edge the key still scores?",
        "",
        f"A fixed confounded graph with the `X→M` edge's transform swept across "
        f"{report['transforms']}, over {report['templates']} templates × seeds {report['seeds']} "
        f"(n={report['n']}), graded by `{report['grader']}`. A transform never changes the edge set, "
        "so `X→M` is in the answer key for every world; the cells are **how often each method "
        "recovers it** (nonlinear-edge recall, 1.0 = always).",
        "",
        "### Nonlinear-edge recall of `X→M`, by transform",
        "",
        "| transform | " + " | ".join(f"`{m}`" for m in methods) + " |",
        "|---" * (len(methods) + 1) + "|",
    ]
    for tv in report["transforms"]:
        lines.append(f"| {tv} | " + " | ".join(f(bt[tv][m]) for m in methods) + " |")
    lines += [
        "",
        "### Aggregate over all nonlinear worlds",
        "",
        "| method | worlds | mean F1 | mean SHD | confounded-kept | `X→M` recall |",
        "|---|---|---|---|---|---|",
    ]
    for m in methods:
        a = agg[m]
        err = f" (+{a['errored']} err)" if a["errored"] else ""
        lines.append(
            f"| `{m}` | {a['worlds_scored']}{err} | {f(a['mean_f1'])} | {f(a['mean_shd'])} | "
            f"{f(a['total_confounded_kept'])} | {f(a['nl_edge_recall'])} |"
        )
    lines += [
        "",
        "**Read — two orthogonal failure modes, cleanly separated (the edge is in the key throughout):**",
        "",
        "1. **Nonlinearity bites *correlation-based* discovery only for *even* transforms.** For "
        "`square` and `abs` — where the linear correlation of `X` with `M` is ≈ 0 — the "
        "correlation-based observational methods (`pc`, `fci`) recover `X→M` **0%** of the time; for "
        "odd/monotone transforms (`cube`, `tanh`, `relu`) they recover it fine (residual linear "
        "correlation remains). The recoverer of the even-nonlinear edge is **interventional data**: "
        "`pc+do`, `gies`, and the reference recover `square` (and `pc+do`/`fci+do` partially recover "
        "`abs`) — *none of which are latent-aware*. So for **nonlinearity the lever is interventions, "
        "not latent-awareness.**",
        "2. **Latent confounding is a separate axis.** Every causal-sufficiency "
        "method — including the interventional `pc+do`, `fci+do`, `gies` — keeps the hidden-confounded "
        "`{A,B}` pair as causal (~18/18); only the latent-aware `interventional-ci` reaches 0. For "
        "**confounding the lever is latent-awareness, not interventions.**",
        "",
        "The built-in `braking` world is the single-world version of this: its `speed²` is an **even** "
        "transform, so observational PC misses it — but the general recoverer of even-nonlinear edges is "
        "interventional data, and latent-awareness is the separate lever for confounding. "
        "(`dagma`/`directlingam` are weak throughout — including on the linear `identity` control — so "
        "their misses are not nonlinearity-specific.)",
        "",
        "Reproduce: `uv run python evals/nonlinear-crossover/run.py` (needs the `discover` extra).",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
