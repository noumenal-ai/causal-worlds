"""Tests for the WorldBrief completeness checklist and prose rendering (pure)."""

from causal_worlds.brief import WorldBrief, is_complete, missing_fields, render


def _full() -> WorldBrief:
    return WorldBrief(
        domain="a coffee chain",
        variables=(
            "price (controllable): shelf price",
            "foot (observable): footfall",
            "sales (outcome): revenue",
        ),
        relationships=("price -> sales: price moves revenue",),
    )


def test_empty_brief_is_incomplete_and_lists_every_required_field():
    assert missing_fields(WorldBrief()) == ("domain", "variables", "relationships")
    assert not is_complete(WorldBrief())


def test_too_few_variables_keeps_variables_missing():
    brief = WorldBrief(domain="x", variables=("a (controllable): a",), relationships=("a -> b: y",))
    assert "variables" in missing_fields(brief)


def test_full_brief_is_complete():
    assert missing_fields(_full()) == ()
    assert is_complete(_full())


def test_optional_fields_never_block_completeness():
    # regimes/hidden/objective left empty -> still complete.
    brief = _full()
    assert brief.regimes == ""
    assert brief.hidden == ""
    assert brief.objective == ""
    assert is_complete(brief)


def test_render_includes_domain_variables_and_relationships():
    text = render(_full())
    assert "a coffee chain" in text
    assert "price (controllable): shelf price" in text
    assert "price -> sales" in text
    assert "none specified" in text  # the unspecified optional fields
