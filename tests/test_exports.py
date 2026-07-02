"""Regression: builder factories + Claim types must be importable from the top level (#30/#31)."""

import causal_worlds
from causal_worlds import (
    Claim,
    ClaimError,
    ClaimModel,
    ClaudeAuthor,
    GeminiJudge,
    build_claude_author,
    build_gemini_judge,
)


def test_previously_missing_builders_import_from_top_level():
    # issue #31: defined in author.py / judge.py but were absent from __init__ (import used to fail)
    assert build_claude_author.__name__ == "build_claude_author"
    assert build_gemini_judge.__name__ == "build_gemini_judge"
    assert ClaudeAuthor.__name__ == "ClaudeAuthor"
    assert GeminiJudge.__name__ == "GeminiJudge"


def test_claim_types_are_exported():
    # issue #30: downstream (a UI on generate()) needs these to consume AdmittedWorld.claim
    assert Claim.__name__ == "Claim"
    assert issubclass(ClaimError, causal_worlds.SpecError)
    assert ClaimModel.__name__ == "ClaimModel"


def test_new_names_are_declared_in_all():
    for name in (
        "build_claude_author",
        "build_gemini_judge",
        "ClaudeAuthor",
        "GeminiJudge",
        "Claim",
        "ClaimError",
        "ClaimModel",
    ):
        assert name in causal_worlds.__all__
        assert hasattr(causal_worlds, name)
