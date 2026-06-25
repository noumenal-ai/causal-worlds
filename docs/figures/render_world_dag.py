"""Render a built-in world's declared SCM to a PNG for the README (renders on GitHub AND PyPI).

The package ships zero-dependency text renderers (``causal_worlds.to_mermaid`` / ``to_dot``); this is
the one-off image generator for the docs, so it may use matplotlib + networkx (NOT package deps).

    uv run --with matplotlib --with networkx python docs/figures/render_world_dag.py

Outputs docs/figures/coffee_world.png. Re-run after changing the world; commit the PNG. The visual
grammar matches the Mermaid/DOT renderers: controllable=blue lever, outcome=green KPI, observable=grey,
disturbance=amber, hidden confounder=dashed red.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from causal_worlds import worlds

WORLD = "coffee"
OUT = Path(__file__).resolve().parent / f"{WORLD}_world.png"

ROLE_COLOR = {
    "controllable": ("#dbeafe", "#1d4ed8"),
    "outcome": ("#dcfce7", "#15803d"),
    "observable": ("#f1f5f9", "#64748b"),
    "disturbance": ("#fef3c7", "#b45309"),
    "hidden": ("#fee2e2", "#b91c1c"),
}


def _layout(spec, g) -> dict[str, tuple[float, float]]:
    """Manual layered DAG layout: x = topological depth, nodes evenly spread on y within a layer."""
    layers = list(nx.topological_generations(g))
    pos: dict[str, tuple[float, float]] = {}
    for x, names in enumerate(layers):
        ordered = sorted(names)  # stable
        n = len(ordered)
        for i, name in enumerate(ordered):
            y = 0.0 if n == 1 else (i - (n - 1) / 2) * (2.4 / max(n - 1, 1))
            pos[name] = (float(x) * 2.2, y)
    return pos


def main() -> None:
    spec = worlds.get(WORLD)
    var = {v.name: v for v in spec.variables}
    g = nx.DiGraph()
    g.add_nodes_from(v.name for v in spec.variables)
    edges = [
        (t.parent, m.target, t.lag, var[t.parent].hidden)
        for m in spec.mechanisms
        for t in (*m.terms, *(m.regime_terms or ()))
    ]
    g.add_edges_from((p, t) for p, t, *_ in edges)
    pos = _layout(spec, g)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    bw, bh = 0.62, 0.34  # node box half-extents are bw/2, bh/2 in data coords

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for parent, target, lag, hidden in edges:
        ax.annotate(
            "", xy=pos[target], xytext=pos[parent],
            arrowprops={"arrowstyle": "-|>", "color": "#b91c1c" if hidden else "#475569",
                        "lw": 1.5, "linestyle": (0, (4, 3)) if hidden else "-",
                        "shrinkA": 26, "shrinkB": 26, "connectionstyle": "arc3,rad=0.12"},
            zorder=1,
        )
        if lag:
            mx, my = (pos[parent][0] + pos[target][0]) / 2, (pos[parent][1] + pos[target][1]) / 2
            ax.text(mx, my + 0.12, f"lag {lag}", fontsize=8, color="#475569", ha="center", zorder=5)
    for v in spec.variables:
        kind = "hidden" if v.hidden else v.role.value
        fill, stroke = ROLE_COLOR[kind]
        x, y = pos[v.name]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.03",
                linewidth=2, edgecolor=stroke, facecolor=fill,
                linestyle="--" if v.hidden else "-", zorder=3,
            )
        )
        ax.text(x, y, v.name, ha="center", va="center", fontsize=10, color=stroke,
                fontweight="bold", zorder=4)

    legend = [mpatches.Patch(facecolor=f, edgecolor=s, label=k) for k, (f, s) in ROLE_COLOR.items()]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.02),
              ncol=5, frameon=False, fontsize=9)
    ax.set_title(f'causal-worlds — the declared SCM for "{WORLD}"  '
                 "(hidden confounder, dashed, is never in the data)",
                 fontsize=11.5, color="#1b1f24", pad=26)
    ax.set_xlim(min(xs) - bw, max(xs) + bw)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.axis("off")
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
