"""Pre-registered reward function for RL congestion control.

r_t = w_g * log2(1 + g_t)                    useful-throughput term
    - w_qd * min(qd_t, 200 ms) / 20 ms       queueing-delay penalty
    - w_loss * min(L_t, 0.05) / 0.01         loss-signal penalty
    - w_act * |log2(action multiplier)|      control-oscillation penalty

where g_t is this interval's goodput (Mbps), qd_t the mean queueing delay
(ms), and L_t the sender-observable loss signal (retransmission-event rate;
see features.IntervalStats.loss_signal). All terms are computed from
sender-observable quantities only, so the same reward is deployable in a
real stack. Weights and normalizations (fixed before any training or test
evaluation; see configs/evaluation_protocol.yaml):

- w_g = 1.0: log-scale throughput so the term is scale-free across link
  rates (10 -> 20 Mbps is worth the same as 1 -> 2 Mbps).
- w_qd = 0.5: 20 ms of standing queueing delay costs as much as halving
  throughput; 40 ms as much as a 4x throughput drop.
- w_loss = 1.0 with a 1%-per-interval reference: a retransmission-event
  rate of 1% costs -1.0 (comparable to a large throughput cut). The signal
  is capped at 5% so a single burst cannot dominate the episode return.
  (An earlier draft counted duplicate ACKs as loss; one lost packet
  generates up to a window's worth of dupacks, which saturated the penalty
  at -10. Corrected before training.)
- w_act = 0.05: mild penalty for large multiplicative jumps to discourage
  oscillatory control; small enough not to block genuine adaptation.

Ablations (see configs/evaluation_protocol.yaml) set w_qd or w_loss to 0.
"""
from __future__ import annotations

import math
from typing import Dict

from ..environment.features import IntervalStats

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_goodput": 1.0,
    "w_queue_delay": 0.5,
    "w_loss": 1.0,
    "w_action": 0.05,
}

LOSS_SIGNAL_CAP = 0.05
LOSS_SIGNAL_REF = 0.01
QD_CAP_MS = 200.0
QD_REF_MS = 20.0


def compute_reward(stats: IntervalStats, action_mult: float,
                   weights: Dict[str, float] = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    g = stats.rate_mbps
    qd = stats.qd_mean_ms if stats.qd_mean_ms is not None else 0.0
    L = stats.loss_signal
    r = w["w_goodput"] * math.log2(1.0 + g)
    r -= w["w_queue_delay"] * min(max(qd, 0.0), QD_CAP_MS) / QD_REF_MS
    r -= w["w_loss"] * min(max(L, 0.0), LOSS_SIGNAL_CAP) / LOSS_SIGNAL_REF
    if action_mult is not None and action_mult > 0 and action_mult != 1.0:
        r -= w["w_action"] * abs(math.log2(action_mult))
    return float(r)
