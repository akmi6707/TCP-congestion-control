"""CUBIC (RFC 8312/9438-style, simplified for the cwnd-only abstraction).

β = 0.7, C = 0.4 (1-segment units scale: C is per second^3 in the RFC's
scaled form; here K = cbrt(W_max*(1-beta)/C) in seconds with cwnd in
packets). TCP-friendly estimate in the convex region follows RFC 8312:
W_tcp(t) = W_loss + 3*(1-beta)/(2-beta) * t/RTT.
Slow start is used until the first loss / reaching ssthresh.
"""
from __future__ import annotations

import math

from .base import BaseController

BETA = 0.7
C_CUBIC = 0.4


class CubicController(BaseController):
    name = "cubic"

    def __init__(self, flow_id: int = 0) -> None:
        super().__init__(flow_id)
        self.w_max = 0.0        # window before last reduction
        self.w_loss = 0.0       # cwnd after last reduction (W_tcp origin)
        self.k = 0.0            # cubic time constant, seconds
        self.epoch_start_ms: float | None = None
        self.last_t_ms = 0.0
        self.in_slow_start = True

    def _reduce(self, flow) -> None:
        self.w_max = max(flow.cwnd, 2.0)
        flow.ssthresh = max(flow.cwnd * BETA, 2.0)
        flow.cwnd = flow.ssthresh
        self.w_loss = flow.cwnd
        self.k = (self.w_max * (1.0 - BETA) / C_CUBIC) ** (1.0 / 3.0) if self.w_max > 0 else 0.0
        self.epoch_start_ms = self.last_t_ms
        self.in_slow_start = False

    def on_fast_retransmit(self, flow) -> None:
        self._reduce(flow)

    def on_rto(self, flow) -> None:
        self._reduce(flow)
        flow.cwnd = 1.0
        self.in_slow_start = True

    def on_tick(self, flow, obs: dict) -> None:
        self.last_t_ms = obs["t_ms"]

    def on_new_acks(self, flow, n_new: int) -> None:
        if self.in_slow_start and flow.cwnd < flow.ssthresh:
            flow.cwnd += n_new
            return
        self.in_slow_start = False
        if self.epoch_start_ms is None:
            # first congestion-avoidance update after startup
            self.w_max = max(flow.cwnd, 2.0)
            self.w_loss = flow.cwnd
            self.epoch_start_ms = self.last_t_ms
        t_s = (self.last_t_ms - self.epoch_start_ms) / 1000.0
        rtt_s = (flow.srtt or 50.0) / 1000.0
        rtt_s = max(rtt_s, 0.005)
        tk = t_s - self.k
        target = C_CUBIC * tk ** 3 + self.w_max if self.w_max > 0 else flow.cwnd
        w_tcp = self.w_loss + 3.0 * (1.0 - BETA) / (2.0 - BETA) * t_s / rtt_s
        cand = max(target, w_tcp) if t_s >= self.k else target
        # clamp per-ACK growth to at most 1/2 of current window (MSS accounting)
        cand = min(cand, flow.cwnd * 1.5 + n_new / max(flow.cwnd, 1.0))
        flow.cwnd = max(2.0, cand)
