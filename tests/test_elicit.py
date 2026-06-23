"""Tests for the elicitation Session loop, driven by a scripted FakeElicitor (keyless)."""

from causal_worlds.brief import WorldBrief, is_complete
from causal_worlds.elicit import force_ready, respond, start_session
from causal_worlds.fakes import FakeElicitor
from causal_worlds.protocols import Elicitor

_PARTIAL = WorldBrief(domain="a coffee chain", variables=("price (controllable): price",))
_FULL = WorldBrief(
    domain="a coffee chain",
    variables=(
        "price (controllable): price",
        "foot (observable): footfall",
        "sales (outcome): rev",
    ),
    relationships=("price -> sales: price moves revenue",),
)


def test_fake_elicitor_satisfies_the_protocol():
    assert isinstance(FakeElicitor(steps=[(_FULL, None)]), Elicitor)


def test_session_starts_unready_with_an_opening_question():
    session = start_session()
    assert not session.ready
    assert session.question is not None
    assert session.transcript[0][0] == "assistant"


def test_dialogue_advances_then_completes_when_elicitor_signals_ready():
    # turn 1: ask a follow-up (partial brief); turn 2: ready (full brief, no question).
    elicitor = FakeElicitor(steps=[(_PARTIAL, "what's the outcome?"), (_FULL, None)])
    session = start_session()

    session = respond(elicitor, session, "a coffee chain, I can set price")
    assert not session.ready
    assert session.question == "what's the outcome?"
    assert session.brief == _PARTIAL

    session = respond(elicitor, session, "footfall and sales matter")
    assert session.ready
    assert session.question is None
    assert is_complete(session.brief)
    # transcript grew by two turns (user + assistant) each round, on top of the opening.
    assert [role for role, _ in session.transcript] == [
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_force_ready_hands_off_even_when_incomplete():
    elicitor = FakeElicitor(steps=[(_PARTIAL, "more?")])
    session = respond(elicitor, start_session(), "a coffee chain")
    assert not session.ready  # still mid-dialogue
    forced = force_ready(session)
    assert forced.ready
    assert forced.question is None
    assert not is_complete(forced.brief)  # forced through with a thin brief
