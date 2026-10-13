"""Control-interval aggregation and feature extraction for learning controllers.

Learning controllers act every ``interval_ticks`` simulator ticks (default
5 ticks = 25 ms). This module aggregates per-tick sender-observable
observations into interval statistics and builds the documented feature
vector. Everything is computed from sender-visible information only (no
bottleneck-internal state), and all features are clipped to documented
ranges; missing values (e.g. no RTT sample yet) mark the vector invalid so
callers can bootstrap safely.

RL-only feature vector (18 dims, indices fixed; used by the RL-only
controller; the hybrid appends 3 ML predictions):

  idx name              definition                                   clip
  0   rate_now          log2(1 + goodput this interval, Mbps)        [0, 25]
  1   rate_prev         log2(1 + previous interval goodput, Mbps)     [0, 25]
  2   bw_est            log2(1 + 1s delivered-rate estimate, Mbps)     [0, 25]
  3   min_rtt           log2(min_rtt_ms / 5.0)                        [-1, 7]
  4   qd                mean queueing delay this interval / 20 ms     [0, 10]
  5   qd_ratio          srtt/min_rtt - 1 at interval end              [0, 5]
  6   cwnd              log2(1 + cwnd, packets)                       [0, 20]
  7   bdp_est           log2(1 + BDP estimate, packets)               [0, 20]
  8   cwnd_over_bdp     cwnd / BDP estimate - 1                       [-1, 5]
  9   inflight_frac     inflight / max(cwnd, 1)                        [0, 1.2]
  10  loss_flag         1 if any loss event (fast retx) in interval    {0,1}
  11  dup_frac          dupacks / max(delivered pkts, 1)               [0, 10]
  12  rto_flag          1 if any RTO in interval                       {0,1}
  13  utilization       rate_now / max(bw_est, 0.1) - 1                [-1, 1]
  14  rate_trend        (rate_now - rate_prev)/max(rate_prev, 0.1)    [-1, 1]
  15  qd_trend          (qd - qd_prev)/20 ms                           [-1, 1]
  16  prev_action       previous action index / (n_actions-1)          [0, 1]
  17  prev_reward       previous reward / 5                            [-1, 1]

Hybrid additions (predictions of the CURRENT interval, from the ML model,
computed at the control point BEFORE the action is chosen):
  18  pred_qd           predicted mean queueing delay / 20 ms          [0, 10]
  19  pred_loss_pct    predicted loss fraction x 100 (percent)         [0, 10]
  20  pred_rate         log2(1 + predicted goodput, Mbps)              [0, 25]
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

PAYLOAD_BYTES = 1460.0
N_BASE_FEATURES = 18
N_HYBRID_FEATURES = 3
FEATURE_NAMES: List[str] = [
    "rate_now", "rate_prev", "bw_est", "min_rtt", "qd", "qd_ratio",
    "cwnd", "bdp_est", "cwnd_over_bdp", "inflight_frac", "loss_flag",
    "dup_frac", "rto_flag", "utilization", "rate_trend", "qd_trend",
    "prev_action", "prev_reward",
]
HYBRID_FEATURE_NAMES: List[str] = ["pred_qd", "pred_loss_pct", "pred_rate"]


def _clip(x: float, lo: float, hi: float) -> float:
    if x is None or not math.isfinite(x):
        return lo
    return float(min(max(x, lo), hi))


class IntervalStats:
    """Aggregated statistics of one control interval (sender-observable)."""

    __slots__ = ("t_start_ms", "t_end_ms", "delivered_pkts", "dupacks",
                 "loss_events", "rto_events", "srtt_mean_ms", "srtt_end_ms",
                 "min_rtt_end_ms", "cwnd_end", "inflight_end", "bw_est_mbps",
                 "ticks")

    def __init__(self):
        self.t_start_ms = 0.0
        self.t_end_ms = 0.0
        self.delivered_pkts = 0
        self.dupacks = 0
        self.loss_events = 0
        self.rto_events = 0
        self.srtt_mean_ms = None
        self.srtt_end_ms = None
        self.min_rtt_end_ms = None
        self.cwnd_end = 0.0
        self.inflight_end = 0
        self.bw_est_mbps = None
        self.ticks = 0

    @property
    def rate_mbps(self) -> float:
        dur_s = max((self.t_end_ms - self.t_start_ms) / 1000.0, 1e-6)
        return self.delivered_pkts * PAYLOAD_BYTES * 8.0 / dur_s / 1e6

    @property
    def qd_mean_ms(self) -> Optional[float]:
        if self.srtt_mean_ms is None or self.min_rtt_end_ms is None:
            return None
        return max(0.0, self.srtt_mean_ms - self.min_rtt_end_ms)

    @property
    def qd_end_ms(self) -> Optional[float]:
        if self.srtt_end_ms is None or self.min_rtt_end_ms is None:
            return None
        return max(0.0, self.srtt_end_ms - self.min_rtt_end_ms)

    @property
    def bdp_est_pkts(self) -> float:
        if self.bw_est_mbps is None or self.min_rtt_end_ms is None:
            return 0.0
        return max(self.bw_est_mbps * 1e6 / 8.0 / PAYLOAD_BYTES *
                   (self.min_rtt_end_ms / 1000.0), 1.0)

    @property
    def loss_signal(self) -> float:
        """Sender-observable loss intensity: retransmission-event rate.

        Counts fast-retransmit + RTO triggers (each corresponds to a lost
        segment needing repair) normalized by delivered packets. Duplicate
        ACKs are NOT counted: one lost packet generates up to a window's
        worth of dupacks, so including them saturates the signal (found
        during pre-training calibration); dupacks remain a separate feature.
        """
        num = self.loss_events + self.rto_events
        return num / max(self.delivered_pkts, 1)


class IntervalAggregator:
    """Per-flow per-tick observation aggregation into control intervals."""

    def __init__(self, interval_ticks: int = 5, dt_ms: float = 5.0) -> None:
        self.interval_ticks = max(1, int(interval_ticks))
        self.dt_ms = dt_ms
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._stats = IntervalStats()
        self.prev_stats: Optional[IntervalStats] = None
        self.last_t_ms = 0.0

    def on_tick(self, obs: dict) -> None:
        """Accumulate one tick; returns True when an interval just closed."""
        st = self._stats
        if st.ticks == 0:
            st.t_start_ms = obs["t_ms"]
        st.t_end_ms = obs["t_ms"] + self.dt_ms
        st.ticks += 1
        st.delivered_pkts += obs.get("new_acks") or 0
        st.dupacks += obs.get("dupacks") or 0
        if obs.get("loss_event"):
            st.loss_events += 1
        if obs.get("rto_event"):
            st.rto_events += 1
        if obs.get("srtt_ms") is not None:
            if st.srtt_mean_ms is None:
                st.srtt_mean_ms = obs["srtt_ms"]
            else:
                # running mean over ticks with a sample
                st.srtt_mean_ms += (obs["srtt_ms"] - st.srtt_mean_ms) / st.ticks
        st.srtt_end_ms = obs.get("srtt_ms")
        st.min_rtt_end_ms = obs.get("min_rtt_ms")
        st.cwnd_end = obs.get("cwnd") or 0.0
        st.inflight_end = obs.get("inflight") or 0
        st.bw_est_mbps = obs.get("est_bw_mbps")
        self._count += 1
        self.last_t_ms = obs["t_ms"]

    def interval_ready(self) -> bool:
        return self._count >= self.interval_ticks

    def take_stats(self) -> IntervalStats:
        """Close the current interval and start a new one."""
        st = self._stats
        self.prev_stats = st
        self._stats = IntervalStats()
        self._count = 0
        return st

    def has_history(self) -> bool:
        return self.prev_stats is not None


def build_features(stats: IntervalStats, prev: Optional[IntervalStats],
                   prev_action_idx: int, prev_reward: float,
                   n_actions: int) -> Optional[np.ndarray]:
    """RL-only feature vector at a control point (start of the next interval).

    ``stats`` is the just-closed interval; ``prev`` the one before it.
    Returns None if the observation is not yet valid (no RTT sample).
    """
    if stats.srtt_end_ms is None or stats.min_rtt_end_ms is None \
            or stats.ticks == 0 or stats.delivered_pkts == 0:
        return None
    prev_rate = prev.rate_mbps if prev is not None else stats.rate_mbps
    qd = stats.qd_mean_ms if stats.qd_mean_ms is not None else 0.0
    qd_prev = prev.qd_mean_ms if (prev is not None and prev.qd_mean_ms is not None) else qd
    bw = stats.bw_est_mbps if stats.bw_est_mbps is not None else stats.rate_mbps
    bdp = stats.bdp_est_pkts
    cwnd = max(stats.cwnd_end, 1.0)
    f = np.zeros(N_BASE_FEATURES, dtype=np.float32)
    f[0] = _clip(math.log2(1.0 + stats.rate_mbps), 0.0, 25.0)
    f[1] = _clip(math.log2(1.0 + prev_rate), 0.0, 25.0)
    f[2] = _clip(math.log2(1.0 + bw), 0.0, 25.0)
    f[3] = _clip(math.log2(max(stats.min_rtt_end_ms, 1.0) / 5.0), -1.0, 7.0)
    f[4] = _clip(qd / 20.0, 0.0, 10.0)
    f[5] = _clip((stats.srtt_end_ms / max(stats.min_rtt_end_ms, 1.0)) - 1.0, 0.0, 5.0)
    f[6] = _clip(math.log2(1.0 + cwnd), 0.0, 20.0)
    f[7] = _clip(math.log2(1.0 + bdp), 0.0, 20.0)
    f[8] = _clip(cwnd / max(bdp, 1.0) - 1.0, -1.0, 5.0)
    f[9] = _clip(stats.inflight_end / max(cwnd, 1.0), 0.0, 1.2)
    f[10] = 1.0 if stats.loss_events > 0 else 0.0
    f[11] = _clip(stats.dupacks / max(stats.delivered_pkts, 1), 0.0, 10.0)
    f[12] = 1.0 if stats.rto_events > 0 else 0.0
    f[13] = _clip(stats.rate_mbps / max(bw, 0.1) - 1.0, -1.0, 1.0)
    f[14] = _clip((stats.rate_mbps - prev_rate) / max(prev_rate, 0.1), -1.0, 1.0)
    f[15] = _clip((qd - qd_prev) / 20.0, -1.0, 1.0)
    f[16] = _clip(prev_action_idx / max(n_actions - 1, 1), 0.0, 1.0)
    f[17] = _clip(prev_reward / 5.0, -1.0, 1.0)
    if not np.all(np.isfinite(f)):
        return None
    return f


def append_hybrid_predictions(features: np.ndarray, pred_qd_ms: float,
                              pred_loss_frac: float,
                              pred_rate_mbps: float) -> np.ndarray:
    """Append the 3 ML predictions to an RL-only feature vector."""
    out = np.zeros(N_BASE_FEATURES + N_HYBRID_FEATURES, dtype=np.float32)
    out[:N_BASE_FEATURES] = features
    out[N_BASE_FEATURES + 0] = _clip(pred_qd_ms / 20.0, 0.0, 10.0)
    out[N_BASE_FEATURES + 1] = _clip(pred_loss_frac * 100.0, 0.0, 10.0)
    out[N_BASE_FEATURES + 2] = _clip(math.log2(1.0 + max(pred_rate_mbps, 0.0)), 0.0, 25.0)
    return out
