"""Unit tests for feature extraction and normalization."""
import math

import numpy as np
import pytest

from src.environment.features import (IntervalAggregator, build_features,
                                      append_hybrid_predictions, IntervalStats,
                                      N_BASE_FEATURES, FEATURE_NAMES)


def _tick_obs(t_ms, new_acks=5, srtt=30.0, min_rtt=25.0, cwnd=50.0,
              inflight=40, est_bw=4.0, dupacks=0, loss=False, rto=False):
    return {"tick": int(t_ms / 5), "t_ms": t_ms, "cwnd": cwnd,
            "inflight": inflight, "ssthresh": None, "srtt_ms": srtt,
            "min_rtt_ms": min_rtt, "est_bw_mbps": est_bw,
            "new_acks": new_acks, "dupacks": dupacks, "dupacks_pending": 0,
            "loss_event": loss, "rto_event": rto, "done_sending": False,
            "is_short": False}


def test_aggregator_interval_boundary():
    ag = IntervalAggregator(interval_ticks=5, dt_ms=5.0)
    for i in range(4):
        ag.on_tick(_tick_obs(i * 5.0))
        assert not ag.interval_ready()
    ag.on_tick(_tick_obs(20.0))
    assert ag.interval_ready()
    st = ag.take_stats()
    assert st.delivered_pkts == 25  # 5 ticks x 5 acks
    assert not ag.interval_ready()


def test_features_valid_and_clipped():
    ag = IntervalAggregator(interval_ticks=5, dt_ms=5.0)
    for i in range(5):
        ag.on_tick(_tick_obs(i * 5.0, srtt=1e6, min_rtt=1.0, est_bw=1e6))
    st = ag.take_stats()
    f = build_features(st, None, prev_action_idx=0, prev_reward=0.0, n_actions=8)
    assert f is not None and f.shape == (N_BASE_FEATURES,)
    assert np.all(np.isfinite(f))
    # qd feature (idx 4) clipped at 10: (1e6-1)/20 >> 10
    assert f[4] == 10.0
    # rate feature: 25 pkts delivered over 25 ms
    # rate = 25 * 1460 * 8 / 0.025s / 1e6 = 11.68 Mbps
    assert abs(f[0] - math.log2(1 + 11.68)) < 1e-3
    # huge dup_frac clipped
    assert f[11] <= 10.0


def test_features_invalid_without_rtt():
    ag = IntervalAggregator(interval_ticks=5, dt_ms=5.0)
    for i in range(5):
        ag.on_tick(_tick_obs(i * 5.0, srtt=None, min_rtt=None))
    st = ag.take_stats()
    assert build_features(st, None, 0, 0.0, 8) is None


def test_features_invalid_without_delivery():
    ag = IntervalAggregator(interval_ticks=5, dt_ms=5.0)
    for i in range(5):
        ag.on_tick(_tick_obs(i * 5.0, new_acks=0))
    st = ag.take_stats()
    assert build_features(st, None, 0, 0.0, 8) is None


def test_hybrid_append():
    f = np.arange(N_BASE_FEATURES, dtype=np.float32)
    out = append_hybrid_predictions(f, 1000.0, 1.0, 1e6)
    assert out.shape == (N_BASE_FEATURES + 3,)
    assert out[N_BASE_FEATURES + 0] == 10.0      # qd 1000/20 clipped
    assert out[N_BASE_FEATURES + 1] == 10.0      # loss 100 pct clipped
    # log2(1 + 1e6) ~ 19.93 (below the 25 clip)
    assert out[N_BASE_FEATURES + 2] == pytest.approx(math.log2(1 + 1e6), rel=1e-4)
    out2 = append_hybrid_predictions(f, 1000.0, 1.0, 1e9)
    assert out2[N_BASE_FEATURES + 2] == 25.0     # 1e9 Mbps clipped at 25
    assert np.all(np.isfinite(out))


def test_loss_signal_definition():
    st = IntervalStats()
    st.delivered_pkts = 100
    st.loss_events = 1
    st.rto_events = 0
    st.dupacks = 50  # dupacks must NOT count as loss
    assert abs(st.loss_signal - 0.01) < 1e-9


def test_feature_names_match_dims():
    assert len(FEATURE_NAMES) == N_BASE_FEATURES
