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

from causal_worlds.schema import Role, Variable, WorldSpec


@dataclass(frozen=True, slots=True)
class _Edge:
    """One directed edge for rendering: parent → target, with its path coefficients and lags."""

    parent: str
    target: str
    lags: frozenset[int]
    coeffs: tuple[float, ...]  # the path coefficient(s); two distinct ⇒ a regime sign-flip
    regime_only: bool


def _collect_edges(spec: WorldSpec) -> list[_Edge]:
    """Flatten mechanisms into deduped directed edges, keeping each edge's coefficients + lags."""
    lags: dict[tuple[str, str], set[int]] = {}
    coeffs: dict[tuple[str, str], list[float]] = {}
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
            # an edge is "regime-only" iff every term that produced it came from regime_terms
            regime_only[edge] = regime_only.get(edge, True) and is_regime
    return [
        _Edge(p, t, frozenset(edge_lags), tuple(coeffs[(p, t)]), regime_only[(p, t)])
        for (p, t), edge_lags in lags.items()
    ]


def _edge_label(edge: _Edge) -> str:
    """The edge label: the Wright **path coefficient(s)**, plus any lag.

    A single coefficient (``0.8``) is the usual case; two (``-1/1``) mean the effect *flips* under a
    regime — the anti-cliché lever. ``lag k`` marks a temporal edge.
    """
    coeff = "/".join(f"{c:g}" for c in edge.coeffs)
    nonzero = sorted(lag for lag in edge.lags if lag > 0)
    lag = (" lag " + ",".join(str(lag) for lag in nonzero)) if nonzero else ""
    return coeff + lag


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


def _mermaid_node(variable: Variable) -> str:
    if variable.hidden:
        return f'    {_ident(variable.name)}(("{variable.name}")):::hidden'
    open_, close = _MERMAID_SHAPE[variable.role]
    return f'    {_ident(variable.name)}{open_}"{variable.name}"{close}:::{variable.role.value}'


def to_mermaid(spec: WorldSpec) -> str:
    """Render ``spec`` as a Mermaid ``graph LR`` flowchart (renders natively on GitHub).

    Args:
        spec: The world to draw. Hidden confounders are drawn as dashed circles.

    Returns:
        A Mermaid diagram string, ready to drop in a fenced ``mermaid`` Markdown block.
    """
    hidden = {v.name for v in spec.variables if v.hidden}
    lines = ["graph LR"]
    lines += [_mermaid_node(v) for v in spec.variables]
    for edge in _collect_edges(spec):
        arrow = "-.->" if (edge.regime_only or edge.parent in hidden) else "-->"
        lines.append(
            f'    {_ident(edge.parent)} {arrow}|"{_edge_label(edge)}"| {_ident(edge.target)}'
        )
    lines += _MERMAID_CLASSDEFS
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


def _dot_node(variable: Variable) -> str:
    attrs = _DOT_HIDDEN if variable.hidden else _DOT_STYLE[variable.role]
    return f'  "{variable.name}" [label="{variable.name}", {attrs}];'


def to_dot(spec: WorldSpec) -> str:
    """Render ``spec`` as Graphviz DOT (``digraph``), for ``dot``/any Graphviz viewer.

    Args:
        spec: The world to draw. Hidden confounders are drawn as dashed circles.

    Returns:
        A DOT ``digraph`` string; render with e.g. ``dot -Tpng world.dot -o world.png``.
    """
    hidden = {v.name for v in spec.variables if v.hidden}
    lines = [
        'digraph "causal-worlds" {',
        "  rankdir=LR;",
        '  bgcolor="white";',
        '  graph [fontname="Helvetica"];',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=11, color="#475569"];',
    ]
    lines += [_dot_node(v) for v in spec.variables]
    for edge in _collect_edges(spec):
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
