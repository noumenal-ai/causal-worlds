"""Tests for the observability seam (Tracer / NullTracer / LangfuseTracer), all keyless."""

from causal_worlds import worlds
from causal_worlds.config import Settings
from causal_worlds.container import build_container
from causal_worlds.discover import InterventionalCiDiscoverer
from causal_worlds.fakes import FakeAuthor
from causal_worlds.generate import generate
from causal_worlds.obs import LangfuseTracer, NullTracer, Tracer

_FAST = InterventionalCiDiscoverer(n=4000)


class _RecordingTracer:
    """A Tracer that records the span names + recorded outputs (to prove instrumentation fires)."""

    def __init__(self):
        self.spans = []
        self.outputs = []
        self.flushed = False

    def span(self, name, **inputs):
        self.spans.append(name)
        return NullTracer().span(name, **inputs)  # reuse the no-op context manager

    def record(self, **outputs):
        self.outputs.append(outputs)

    def flush(self):
        self.flushed = True


class _FakeLangfuseClient:
    """Duck-types the Langfuse methods LangfuseTracer uses."""

    def __init__(self):
        self.observations = []
        self.updates = []
        self.flushed = False

    def start_as_current_observation(self, *, name, as_type):
        self.observations.append((name, as_type))
        return NullTracer().span(name)

    def update_current_span(self, **kwargs):
        self.updates.append(kwargs)

    def flush(self):
        self.flushed = True


def test_nulltracer_is_a_tracer_and_no_ops():
    assert isinstance(NullTracer(), Tracer)
    with NullTracer().span("anything", x=1):
        NullTracer().record(y=2)  # must not raise


def test_langfuse_tracer_sets_input_output_and_flushes():
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(client)
    assert isinstance(tracer, Tracer)
    with tracer.span("generate", prompt="a webshop"):
        tracer.record(admitted=True)
    assert client.observations == [("generate", "span")]
    assert {"input": {"prompt": "a webshop"}} in client.updates
    assert {"output": {"admitted": True}} in client.updates
    tracer.flush()
    assert client.flushed


def test_generate_wraps_steps_in_spans_and_records_outputs():
    tracer = _RecordingTracer()
    generate(
        "a webshop", author=FakeAuthor([worlds.get("ecommerce")]), discoverer=_FAST, tracer=tracer
    )
    assert {"generate", "author", "gate"} <= set(tracer.spans)
    assert any(o.get("admitted") for o in tracer.outputs)  # an admit outcome was recorded


def test_container_defaults_to_null_tracer():
    tracer = build_container(Settings(langfuse_enabled=False)).tracer()
    assert isinstance(tracer, NullTracer)


def test_container_builds_langfuse_when_enabled(monkeypatch):
    sentinel = LangfuseTracer(_FakeLangfuseClient())
    monkeypatch.setattr("causal_worlds.container.build_langfuse_tracer", lambda: sentinel)
    tracer = build_container(Settings(langfuse_enabled=True)).tracer()
    assert tracer is sentinel
