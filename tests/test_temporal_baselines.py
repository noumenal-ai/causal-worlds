"""Tests for temporal scoring and the TS-baseline graph parsers (pure; no TS libs needed)."""

import numpy as np

from causal_worlds import grade_temporal_spec, temporal_answer_key, worlds
from causal_worlds.evaluation import temporal_directed_shd, temporal_f1, temporal_score
from causal_worlds.protocols import TemporalDiscoverer
from causal_worlds.temporal_baselines import (
    TEMPORAL_BASELINES,
    GrangerDiscoverer,
    PcmciPlusDiscoverer,
    VarLingamDiscoverer,
    parse_pcmci_graph,
    parse_varlingam,
)

_NAMES = ("a", "b", "c")


def test_temporal_score_exact_and_lagged():
    truth = frozenset({("a", "b", 0), ("b", "c", 1), ("c", "c", 1)})
    perfect = temporal_score(truth, truth)
    assert perfect.temporal_shd == 0
    assert perfect.temporal_f1 == 1.0
    assert perfect.n_truth == 3

    missing_one = temporal_score(frozenset({("a", "b", 0), ("b", "c", 1)}), truth)
    assert missing_one.temporal_shd == 1  # the autoregressive (c,c,1) edge is missing


def test_temporal_shd_counts_lag0_reversal_once():
    truth = frozenset({("a", "b", 0)})
    reversed_edge = frozenset({("b", "a", 0)})
    assert temporal_directed_shd(reversed_edge, truth) == 1  # a flip, not a miss + an extra


def test_temporal_f1_handles_empty():
    assert temporal_f1(frozenset(), frozenset()) == 1.0
    assert temporal_f1(frozenset({("a", "b", 1)}), frozenset()) == 0.0


def test_parse_pcmci_graph_reads_directed_links():
    # graph[i][j][tau] == '-->' means i ->(tau) j; '<->' is latent confounding (not a causal edge).
    graph = np.full((3, 3, 2), "", dtype="<U3")
    graph[0][1][0] = "-->"  # a -> b contemporaneous
    graph[1][2][1] = "-->"  # b ->(lag 1) c
    graph[0][2][0] = "<->"  # a <-> c latent confounding -> not an edge
    edges = parse_pcmci_graph(graph, _NAMES)
    assert edges == frozenset({("a", "b", 0), ("b", "c", 1)})


def test_parse_varlingam_thresholds_and_drops_lag0_self():
    # matrices[k][j][i] = effect of i ->(k) j.
    mats = np.zeros((2, 3, 3))
    mats[0][1][0] = 0.9  # a -> b contemporaneous
    mats[1][2][1] = 0.5  # b ->(lag 1) c
    mats[0][0][0] = 0.8  # a -> a at lag 0: a self-loop, dropped
    mats[1][0][1] = 0.05  # below threshold, ignored
    edges = parse_varlingam(mats, _NAMES)
    assert edges == frozenset({("a", "b", 0), ("b", "c", 1)})


def test_grade_temporal_spec_scores_against_the_lagged_truth():
    spec = worlds.get("supply")
    truth = temporal_answer_key(spec)

    class _PerfectTemporal:
        def recover_temporal(self, substrate, *, seed):  # noqa: ARG002
            return truth

    report = grade_temporal_spec(spec, _PerfectTemporal())
    assert report.temporal_shd == 0
    assert report.temporal_f1 == 1.0
    assert report.n_truth == len(truth)


def test_temporal_baselines_satisfy_the_protocol():
    assert set(TEMPORAL_BASELINES) == {"pcmci+", "lpcmci", "varlingam", "granger"}
    assert isinstance(PcmciPlusDiscoverer(), TemporalDiscoverer)
    assert isinstance(VarLingamDiscoverer(), TemporalDiscoverer)
    assert isinstance(GrangerDiscoverer(), TemporalDiscoverer)
