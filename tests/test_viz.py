"""Tests for the Mermaid / DOT graph renderers."""

from causal_worlds import to_dot, to_mermaid, worlds
from causal_worlds.schema import Mechanism, Role, Term, Variable, WorldSpec


def _temporal_world() -> WorldSpec:
    """lever ->(lag1) mid -> kpi, with a hidden confounder L -> {lever-side, kpi-side}."""
    return WorldSpec(
        variables=(
            Variable("lever", Role.CONTROLLABLE),
            Variable("mid", Role.OBSERVABLE),
            Variable("kpi", Role.OUTCOME),
            Variable("shock", Role.DISTURBANCE),
            Variable("L", Role.OBSERVABLE, hidden=True),
        ),
        mechanisms=(
            Mechanism("mid", (Term("lever", 0.8, lag=1), Term("L", 0.5))),
            Mechanism("kpi", (Term("mid", 0.9), Term("L", 0.4), Term("shock", 0.2))),
        ),
    )


def test_mermaid_has_header_nodes_and_edges():
    spec = worlds.get("coffee")
    out = to_mermaid(spec)
    assert out.startswith("graph LR")
    for v in spec.variables:
        assert v.name in out  # every variable is drawn (label keeps the original name)
    # at least one declared edge appears as an arrow
    assert "-->" in out or "-.->" in out


def test_mermaid_marks_hidden_lag_and_regime():
    out = to_mermaid(_temporal_world())
    assert ":::hidden" in out  # the hidden confounder gets the hidden class
    assert "((" in out  # ...drawn as a circle
    assert "lag 1" in out  # the lagged edge is labelled
    assert "-.->" in out  # edges out of the hidden node are dashed
    assert ":::controllable" in out  # controllable role styled
    assert ":::outcome" in out  # outcome role styled


def test_dot_is_a_digraph_with_nodes_and_edges():
    out = to_dot(worlds.get("coffee"))
    assert out.startswith('digraph "causal-worlds" {')
    assert out.rstrip().endswith("}")
    assert "rankdir=LR" in out
    assert "->" in out


def test_dot_marks_hidden_and_lag():
    out = to_dot(_temporal_world())
    assert "dashed" in out  # hidden-origin / regime edges (and the hidden node) are dashed
    assert 'label="lag 1"' in out


def test_renderers_handle_names_needing_sanitization():
    # a name with a space/dash must not break the Mermaid node id (label keeps the original)
    spec = WorldSpec(
        variables=(Variable("ad spend", Role.CONTROLLABLE), Variable("rev-day", Role.OUTCOME)),
        mechanisms=(Mechanism("rev-day", (Term("ad spend", 1.0),)),),
    )
    mermaid = to_mermaid(spec)
    assert '"ad spend"' in mermaid  # original label preserved
    assert "ad_spend" in mermaid  # id sanitized
    assert to_dot(spec).count("->") == 1
