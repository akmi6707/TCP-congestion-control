"""Unit tests for the reward function and its edge cases."""
import math

import pytest

from src.environment.features import IntervalStats
from src.models.reward import compute_reward, DEFAULT_WEIGHTS


def _stats(rate_mbps, qd_ms=0.0, n_loss_events=0, rto_events=0, dupacks=0,
           dur_ms=250.0):
    st = IntervalStats()
    st.t_start_ms, st.t_end_ms = 0.0, dur_ms
    # delivered = rate * dur / (1460*8)
    st.delivered_pkts = int(round(rate_mbps * 1e6 * (dur_ms / 1000.0) / (1460 * 8)))
    st.loss_events = n_loss_events
    st.rto_events = rto_events
    st.dupacks = dupacks
    st.srtt_mean_ms = qd_ms + 25.0   # min_rtt 25 -> qd = srtt - 25
    st.min_rtt_end_ms = 25.0
    st.srtt_end_ms = qd_ms + 25.0
    return st


def test_reward_grows_with_rate():
    s1, s10 = _stats(1.0), _stats(10.0)
    r1 = compute_reward(s1, 1.0)
    r10 = compute_reward(s10, 1.0)
    assert r10 > r1
    # log-scale throughput term: difference = w_g * (log2(1+r10) - log2(1+r1))
    expected = DEFAULT_WEIGHTS["w_goodput"] * (
        math.log2(1 + s10.rate_mbps) - math.log2(1 + s1.rate_mbps))
    assert abs((r10 - r1) - expected) < 1e-6


def test_reward_penalty_queue_delay():
    r0 = compute_reward(_stats(5.0, qd_ms=0.0), 1.0)
    r40 = compute_reward(_stats(5.0, qd_ms=40.0), 1.0)
    assert abs((r0 - r40) - 0.5 * 40.0 / 20.0) < 1e-6


def test_reward_loss_cap_and_reference():
    r_small = compute_reward(_stats(5.0, n_loss_events=0), 1.0)
    r_1loss = compute_reward(_stats(5.0, n_loss_events=1), 1.0)
    # 5 Mbps over 250 ms -> ~107 delivered; 1 loss / 107 ~ 0.93% signal
    st = _stats(5.0, n_loss_events=1)
    expected_pen = DEFAULT_WEIGHTS["w_loss"] * min(st.loss_signal, 0.05) / 0.01
    assert abs((r_small - r_1loss) - expected_pen) < 1e-4
    # saturation at 5% signal
    r_10 = compute_reward(_stats(5.0, n_loss_events=10), 1.0)
    r_50 = compute_reward(_stats(5.0, n_loss_events=50), 1.0)
    assert abs(r_10 - r_50) < 1e-6


def test_reward_action_penalty():
    r_hold = compute_reward(_stats(5.0), 1.0)
    r_half = compute_reward(_stats(5.0), 0.5)
    assert abs((r_hold - r_half) - DEFAULT_WEIGHTS["w_action"] * 1.0) < 1e-6


def test_ablation_weights_zero_out_terms():
    w = dict(DEFAULT_WEIGHTS); w["w_queue_delay"] = 0.0
    assert compute_reward(_stats(5.0, qd_ms=100.0), 1.0, w) == \
           pytest.approx(compute_reward(_stats(5.0, qd_ms=0.0), 1.0, w))
    w2 = dict(DEFAULT_WEIGHTS); w2["w_loss"] = 0.0
    assert compute_reward(_stats(5.0, n_loss_events=5), 1.0, w2) == \
           pytest.approx(compute_reward(_stats(5.0, n_loss_events=0), 1.0, w2))


def test_reward_zero_delivery_is_finite():
    st = _stats(0.0)
    st.delivered_pkts = 0
    st.loss_events = 3   # loss with zero delivered: 3/1, capped
    r = compute_reward(st, 1.0)
    assert math.isfinite(r)
    # rate property with 0 delivered -> 0 mbps
    assert st.rate_mbps == 0.0
