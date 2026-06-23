"""Tests for the Claude author and Gemini judge adapters, driven by a fake client (no API key).

The adapters take an injected ``instructor``-style client; here we duck-type one whose
``chat.completions.create`` returns a canned response for the requested ``response_model``. This
exercises all adapter logic — prompt assembly, schema conversion, edge filtering, score clamping —
without importing ``instructor`` or making a network call.
"""

import pytest

from causal_worlds import worlds
from causal_worlds.author import ClaudeAuthor
from causal_worlds.judge import GeminiJudge, _Edge, _Faithfulness, _Prior
from causal_worlds.serde import WorldSpecModel


class _FakeCompletions:
    def __init__(self, responder):
        self._responder = responder
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._responder(kwargs["response_model"], kwargs)


class _FakeChat:
    def __init__(self, responder):
        self.completions = _FakeCompletions(responder)


class _FakeClient:
    def __init__(self, responder):
        self.chat = _FakeChat(responder)


def test_author_converts_model_output_to_spec():
    coffee = worlds.get("coffee")
    client = _FakeClient(lambda _model, _kwargs: WorldSpecModel.from_spec(coffee))
    assert ClaudeAuthor(client, model="claude-test").author("a coffee chain") == coffee


def test_author_complexity_shapes_the_system_brief():
    coffee = worlds.get("coffee")
    captured = {}

    def responder(_model, kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        return WorldSpecModel.from_spec(coffee)

    ClaudeAuthor(_FakeClient(responder), model="t", complexity="easy").author("x")
    assert "NO hidden confounder" in captured["system"]
    ClaudeAuthor(_FakeClient(responder), model="t", complexity="hard").author("x")
    assert "TWO OR MORE hidden confounders" in captured["system"]


def test_author_temporal_mode_adds_the_temporal_clause():
    coffee = worlds.get("coffee")
    captured = {}

    def responder(_model, kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        return WorldSpecModel.from_spec(coffee)

    ClaudeAuthor(_FakeClient(responder), model="t", temporal=True).author("x")
    assert "TEMPORAL operation" in captured["system"]
    assert "autoregressive" in captured["system"]


def test_author_rejects_unknown_complexity():
    with pytest.raises(ValueError, match="unknown complexity"):
        ClaudeAuthor(_FakeClient(lambda _m, _k: None), complexity="nope")


def test_author_threads_feedback_into_the_prompt():
    coffee = worlds.get("coffee")
    client = _FakeClient(lambda _model, _kwargs: WorldSpecModel.from_spec(coffee))
    author = ClaudeAuthor(client, model="claude-test")
    author.author("a coffee chain", feedback="T4 cliché: add a hidden confounder.")
    user_message = client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert "Revise your previous world" in user_message
    assert "hidden confounder" in user_message


def test_judge_prior_filters_unknown_and_self_loops():
    edges = [
        _Edge(src="price", dst="demand"),
        _Edge(src="bogus", dst="demand"),
        _Edge(src="x", dst="x"),
    ]
    client = _FakeClient(lambda _model, _kwargs: _Prior(edges=edges))
    prior = GeminiJudge(client, model="gemini-test").prior_edges(worlds.get("coffee"))
    assert prior == frozenset({("price", "demand")})


def test_judge_faithfulness_is_clamped():
    high = _FakeClient(lambda _model, _kwargs: _Faithfulness(score=1.0, reason="ok"))
    judge = GeminiJudge(high, model="gemini-test")
    assert judge.faithfulness("a coffee chain", worlds.get("coffee")) == 1.0
