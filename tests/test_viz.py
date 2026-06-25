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
    assert "lag 1" in out  # the lagged edge label carries the lag (alongside its coefficient)


def test_renderers_label_edges_with_path_coefficients():
    # Wright's contribution: the number on the wire. lever ->(lag1, 0.8) mid -> (0.9) kpi.
    spec = _temporal_world()
    mermaid, dot = to_mermaid(spec), to_dot(spec)
    assert "0.8 lag 1" in mermaid  # coefficient AND lag on the lagged edge
    assert "0.9" in mermaid  # the mid -> kpi coefficient
    assert "0.8 lag 1" in dot
    assert "0.9" in dot


def test_regime_sign_flip_shows_both_coefficients():
    # coffee's price -> demand is -1.0 normally, +1.0 under the weekend regime: both must show.
    out = to_mermaid(worlds.get("coffee"))
    assert "-1/1" in out  # the sign-flip is visible on the edge label


def test_mermaid_do_cuts_incoming_edges_and_marks_the_forced_node():
    # Rung 2 visualized: do(footfall) cuts every arrow INTO footfall, keeps the arrows OUT.
    surgical = to_mermaid(worlds.get("coffee"), do={"footfall": 1.0})
    assert "footfall = 1" in surgical  # the node is shown set to its value
    assert ":::forced" in surgical  # ...and styled as intervened
    assert "| footfall" not in surgical  # every incoming edge to footfall is cut (surgery)
    assert "footfall -->" in surgical  # but its outgoing effects still flow


def test_dot_do_cuts_incoming_edges():
    surgical = to_dot(worlds.get("coffee"), do={"footfall": 1.0})
    assert "footfall = 1" in surgical
    assert '-> "footfall"' not in surgical  # no edge points into the intervened node
    assert '"footfall" ->' in surgical  # outgoing edges remain


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
