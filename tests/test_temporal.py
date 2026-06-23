"""Tests for the temporal IR, sequential sampling, and the lagged answer-key."""

import numpy as np

from causal_worlds import answer_key, temporal_answer_key, validate, worlds
from causal_worlds.fakes import FakeAuthor, FakeTemporalDiscoverer
from causal_worlds.gates import run_gates
from causal_worlds.generate import generate
from causal_worlds.sample import build_substrate
from causal_worlds.schema import (
    CyclicGraphError,
    Mechanism,
    Role,
    Term,
    Variable,
    WorldSpec,
)


def _ar1() -> WorldSpec:
    # x is an exogenous root; y = 0.7 * y[t-1] + 0.9 * x[t] + noise (autoregressive, stationary).
    return WorldSpec(
        variables=(Variable("x", Role.CONTROLLABLE), Variable("y", Role.OUTCOME)),
        mechanisms=(Mechanism("y", (Term("y", 0.7, lag=1), Term("x", 0.9))),),
    )


def test_lagged_self_loop_is_valid_but_instantaneous_one_is_not():
    validate(_ar1())  # lag-1 self-loop is autoregression — fine

    instantaneous = WorldSpec(
        variables=(Variable("x", Role.CONTROLLABLE), Variable("y", Role.OUTCOME)),
        mechanisms=(Mechanism("y", (Term("y", 0.7, lag=0), Term("x", 0.9))),),
    )
    try:
        validate(instantaneous)
    except CyclicGraphError:
        pass
    else:
        msg = "expected a lag-0 self-loop to be rejected"
        raise AssertionError(msg)


def test_temporal_sampling_is_finite_deterministic_and_autocorrelated():
    substrate = build_substrate(_ar1())
    a = substrate.sample(2000, seed=7)
    b = substrate.sample(2000, seed=7)

    assert a.data.shape == (2000, 2)
    assert np.all(np.isfinite(a.data))
    assert np.array_equal(a.data, b.data)  # deterministic by seed

    # the autoregressive term should make y[t] correlate with y[t-1] (an i.i.d. world would not)
    y = a.data[:, substrate.variables.index("y")]
    lag1_corr = np.corrcoef(y[1:], y[:-1])[0, 1]
    assert lag1_corr > 0.3


def test_summary_answer_key_drops_self_loops_and_collapses_lags():
    key = answer_key(_ar1())
    assert key.edges == frozenset(
        {("x", "y")}
    )  # y->y self-loop excluded; x->y kept (lag collapsed)


def test_temporal_answer_key_keeps_lag_and_self_loops():
    key = temporal_answer_key(_ar1())
    assert ("y", "y", 1) in key  # autoregression retained, with its lag
    assert ("x", "y", 0) in key


def test_builtin_supply_world_is_temporal_and_admits_a_confounded_pair():
    spec = worlds.get("supply")
    assert "supply" in worlds.temporal_names()
    assert "supply" not in worlds.names()  # not in the CLI-gradeable set yet

    substrate = build_substrate(spec)
    sample = substrate.sample(1000, seed=1)
    assert np.all(np.isfinite(sample.data))
    assert "L" not in substrate.variables  # hidden confounder not emitted

    # leadtime and cost share the hidden L with no direct edge -> a confounded pair
    assert frozenset({"leadtime", "cost"}) in answer_key(spec).confounded
    # the lagged truth carries autoregressive self-loops
    assert ("leadtime", "leadtime", 1) in temporal_answer_key(spec)
    assert ("inventory", "inventory", 1) in temporal_answer_key(spec)


def test_temporal_gate_admits_when_ts_reference_recovers_structure():
    spec = worlds.get("supply")
    perfect = FakeTemporalDiscoverer(temporal_answer_key(spec))  # F1 = 1.0 -> admit
    report = run_gates(spec, seed=7, temporal_discoverer=perfect)
    assert report.admitted
    assert report.temporal_grade is not None
    assert report.temporal_grade.temporal_f1 == 1.0


def test_temporal_gate_rejects_when_structure_not_recoverable():
    blind = FakeTemporalDiscoverer(frozenset())  # recovers nothing -> F1 0 -> reject
    report = run_gates(worlds.get("supply"), seed=7, temporal_discoverer=blind)
    assert not report.admitted
    assert "T3 temporal" in report.reason


def test_generate_admits_a_temporal_world_via_the_temporal_gate():
    spec = worlds.get("supply")
    world = generate(
        "a supply operation over time",
        author=FakeAuthor([spec]),
        temporal_discoverer=FakeTemporalDiscoverer(temporal_answer_key(spec)),
        seed=7,
    )
    assert world.report.admitted
    assert world.report.temporal_grade is not None


def test_cross_sectional_worlds_unchanged_by_temporal_path():
    # a world with no lags must still sample i.i.d. (no autocorrelation introduced)
    substrate = build_substrate(worlds.get("ecommerce"))
    data = substrate.sample(3000, seed=3).data
    col = data[:, 0]
    lag1_corr = np.corrcoef(col[1:], col[:-1])[0, 1]
    assert abs(lag1_corr) < 0.1  # i.i.d. rows -> ~no serial correlation
