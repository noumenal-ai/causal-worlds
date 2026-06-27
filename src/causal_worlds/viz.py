"""Render a :class:`WorldSpec` as a graph — zero-dependency Mermaid and Graphviz-DOT exporters.

An SCM *is* a DAG, so the most useful thing you can do with a generated world is look at it. Both
renderers are pure strings (no third-party libraries), so they work in the base install and you can
paste the output straight into a Markdown file (Mermaid renders natively on GitHub) or a Graphviz
viewer (DOT). The renderers deliberately show the **hidden confounders** — the latent nodes a
discovery method never sees are exactly what makes the world hard, so seeing them is the point.

Visual grammar (both renderers):

* **controllable** — a lever you set (stadium / box)
* **outcome** — the KPI you care about (double box)
* **observable** — measured but not set (plain box)
* **disturbance** — an exogenous shock (hexagon / diamond)
* **hidden** — a latent confounder, never emitted as data (dashed circle)

Edges are the generative mechanism's terms (``parent → target``), **labelled with their Wright path
coefficient** — the strength of the cause (what makes it a path diagram, not just a sketch). Two
coefficients (``-1/1``) mean the effect flips under a regime; ``lag k`` marks a temporal edge; an
edge out of a hidden node is dashed (you can't observe its cause).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from causal_worlds.schema import Role, Transform, Variable, WorldSpec

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class _Edge:
    """One directed edge for rendering: parent → target, with its path coefficients and lags."""

    parent: str
    target: str
    lags: frozenset[int]
    coeffs: tuple[float, ...]  # the path coefficient(s); two distinct ⇒ a regime sign-flip
    transforms: frozenset[Transform]  # non-identity nonlinearities on the edge (coeff·f(parent))
    regime_only: bool


def _collect_edges(spec: WorldSpec) -> list[_Edge]:
    """Flatten mechanisms into deduped directed edges, keeping each edge's coefficients + lags."""
    lags: dict[tuple[str, str], set[int]] = {}
    coeffs: dict[tuple[str, str], list[float]] = {}
    transforms: dict[tuple[str, str], set[Transform]] = {}
    regime_only: dict[tuple[str, str], bool] = {}
    for mechanism in spec.mechanisms:
        contemporaneous = [(t, False) for t in mechanism.terms]
        regimes = [(t, True) for t in (mechanism.regime_terms or ())]
        for term, is_regime in contemporaneous + regimes:
            edge = (term.parent, mechanism.target)
            lags.setdefault(edge, set()).add(term.lag)
            seen = coeffs.setdefault(edge, [])
            if term.coeff not in seen:  # keep distinct coeffs in order (base then regime)
                seen.append(term.coeff)
            if term.transform is not Transform.IDENTITY:
                transforms.setdefault(edge, set()).add(term.transform)
            # an edge is "regime-only" iff every term that produced it came from regime_terms
            regime_only[edge] = regime_only.get(edge, True) and is_regime
    return [
        _Edge(
            p,
            t,
            frozenset(edge_lags),
            tuple(coeffs[(p, t)]),
            frozenset(transforms.get((p, t), set())),
            regime_only[(p, t)],
        )
        for (p, t), edge_lags in lags.items()
    ]


def _edge_label(edge: _Edge) -> str:
    """The edge label: the Wright **path coefficient(s)**, any nonlinearity, plus any lag.

    A single coefficient (``0.8``) is the usual case; two (``-1/1``) mean the effect *flips* under a
    regime — the anti-cliché lever. A non-identity transform (``square``) marks an additive-
    nonlinear term — the label then reads ``coeff·f(parent)``, so the coefficient alone is no longer
    the whole effect. ``lag k`` marks a temporal edge.
    """
    coeff = "/".join(f"{c:g}" for c in edge.coeffs)
    fn = (" " + ",".join(sorted(t.value for t in edge.transforms))) if edge.transforms else ""
    nonzero = sorted(lag for lag in edge.lags if lag > 0)
    lag = (" lag " + ",".join(str(lag) for lag in nonzero)) if nonzero else ""
    return coeff + fn + lag


# --------------------------------------------------------------------------- #
# Mermaid
# --------------------------------------------------------------------------- #
_MERMAID_SHAPE = {  # role -> (open, close) brackets; hidden overrides below
    Role.CONTROLLABLE: ("([", "])"),
    Role.OUTCOME: ("[[", "]]"),
    Role.OBSERVABLE: ("[", "]"),
    Role.DISTURBANCE: ("{{", "}}"),
}
_MERMAID_CLASSDEFS = (
    "    classDef controllable fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a;",
    "    classDef outcome fill:#dcfce7,stroke:#15803d,color:#14532d;",
    "    classDef observable fill:#f1f5f9,stroke:#64748b,color:#0f172a;",
    "    classDef disturbance fill:#fef3c7,stroke:#b45309,color:#7c2d12;",
    "    classDef hidden fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d,stroke-dasharray:4 3;",
)


def _ident(name: str) -> str:
    """A safe Mermaid/DOT node id from a variable name (the label keeps the original)."""
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def _mermaid_node(variable: Variable, forced: float | None) -> str:
    if forced is not None:  # an intervened (do) node: shown as set, with its forced value
        return f'    {_ident(variable.name)}["{variable.name} = {forced:g}"]:::forced'
    if variable.hidden:
        return f'    {_ident(variable.name)}(("{variable.name}")):::hidden'
    open_, close = _MERMAID_SHAPE[variable.role]
    return f'    {_ident(variable.name)}{open_}"{variable.name}"{close}:::{variable.role.value}'


def to_mermaid(spec: WorldSpec, *, do: Mapping[str, float] | None = None) -> str:
    """Render ``spec`` as a Mermaid ``graph LR`` flowchart (renders natively on GitHub).

    Args:
        spec: The world to draw. Hidden confounders are drawn as dashed circles.
        do: An optional intervention. Each intervened variable is shown *set* to its value with its
            **incoming edges cut** — i.e. the graph after ``do()`` surgery (Rung 2).

    Returns:
        A Mermaid diagram string, ready to drop in a fenced ``mermaid`` Markdown block.
    """
    forced = dict(do or {})
    hidden = {v.name for v in spec.variables if v.hidden}
    lines = ["graph LR"]
    lines += [_mermaid_node(v, forced.get(v.name)) for v in spec.variables]
    for edge in _collect_edges(spec):
        if edge.target in forced:  # do() surgery: the intervened variable's incoming edges are cut
            continue
        arrow = "-.->" if (edge.regime_only or edge.parent in hidden) else "-->"
        lines.append(
            f'    {_ident(edge.parent)} {arrow}|"{_edge_label(edge)}"| {_ident(edge.target)}'
        )
    lines += _MERMAID_CLASSDEFS
    if forced:
        lines.append(
            "    classDef forced fill:#fde68a,stroke:#b45309,stroke-width:3px,color:#7c2d12;"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Graphviz DOT
# --------------------------------------------------------------------------- #
_DOT_STYLE = {  # role -> graphviz node attributes
    Role.CONTROLLABLE: 'shape=box, style="rounded,filled", fillcolor="#dbeafe", color="#1d4ed8"',
    Role.OUTCOME: 'shape=box, peripheries=2, style=filled, fillcolor="#dcfce7", color="#15803d"',
    Role.OBSERVABLE: 'shape=box, style=filled, fillcolor="#f1f5f9", color="#64748b"',
    Role.DISTURBANCE: 'shape=hexagon, style=filled, fillcolor="#fef3c7", color="#b45309"',
}
_DOT_HIDDEN = 'shape=circle, style="dashed,filled", fillcolor="#fee2e2", color="#b91c1c"'
_DOT_FORCED = 'shape=box, style="filled,bold", fillcolor="#fde68a", color="#b45309", penwidth=2.5'


def _dot_node(variable: Variable, forced: float | None) -> str:
    if forced is not None:  # an intervened (do) node: shown set, with its forced value
        return f'  "{variable.name}" [label="{variable.name} = {forced:g}", {_DOT_FORCED}];'
    attrs = _DOT_HIDDEN if variable.hidden else _DOT_STYLE[variable.role]
    return f'  "{variable.name}" [label="{variable.name}", {attrs}];'


def to_dot(spec: WorldSpec, *, do: Mapping[str, float] | None = None) -> str:
    """Render ``spec`` as Graphviz DOT (``digraph``), for ``dot``/any Graphviz viewer.

    Args:
        spec: The world to draw. Hidden confounders are drawn as dashed circles.
        do: An optional intervention; each intervened variable is shown *set* with its **incoming
            edges cut** — the graph after ``do()`` surgery (Rung 2).

    Returns:
        A DOT ``digraph`` string; render with e.g. ``dot -Tpng world.dot -o world.png``.
    """
    forced = dict(do or {})
    hidden = {v.name for v in spec.variables if v.hidden}
    lines = [
        'digraph "causal-worlds" {',
        "  rankdir=LR;",
        '  bgcolor="white";',
        '  graph [fontname="Helvetica"];',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=11, color="#475569"];',
    ]
    lines += [_dot_node(v, forced.get(v.name)) for v in spec.variables]
    for edge in _collect_edges(spec):
        if edge.target in forced:  # do() surgery: the intervened variable's incoming edges are cut
            continue
        from_hidden = edge.parent in hidden
        attrs = [f'label="{_edge_label(edge)}"']
        if from_hidden or edge.regime_only:
            attrs.append("style=dashed")
        if from_hidden:  # the hidden confounder's influence, in red, so the confounding pops
            attrs.append('color="#b91c1c"')
            attrs.append('fontcolor="#b91c1c"')
        lines.append(f'  "{edge.parent}" -> "{edge.target}" [{", ".join(attrs)}];')
    lines.append("}")
    return "\n".join(lines)
