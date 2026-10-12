"""Delay-based heuristic baseline (Vegas-family, ms-threshold variant).

Classic TCP Vegas measures the backlog diff = (expected - actual) rate x
base_rtt in packets with alpha=2 / beta=4 packet tolerances. In this
simulator's 5 ms-granularity batch model, measured RTT carries a ~1-3 ms
self-drain floor from per-tick ACK-clock clusters; a packet-based diff then
equilibrates far below the BDP at high rates (found in Phase-2 debugging).
This baseline therefore uses the equivalent delay-threshold form: grow the
window while queueing delay (srtt - min_rtt) < alpha_ms, shrink while > beta_ms,
adjusting ~1 packet per RTT. This is the same controller family (delay-driven
AIMD) with thresholds expressed in the delay domain.

- alpha_ms = 5, beta_ms = 15 (fixed, pre-registered)
- bootstrap: additive growth until the first RTT sample exists
- no multiplicative decrease on isolated loss; RTO collapses the window
  (transport requirement)
"""
from __future__ import annotations

from .base import BaseController

ALPHA_MS = 5.0
BETA_MS = 15.0


class VegasLikeController(BaseController):
    name = "vegas_like"

    def __init__(self, flow_id: int = 0) -> None:
        super().__init__(flow_id)
        self.in_slow_start = True

    def on_new_acks(self, flow, n_new: int) -> None:
        if self.in_slow_start and flow.srtt is None:
            flow.cwnd += n_new  # bootstrap growth before any RTT sample

    def on_tick(self, flow, obs: dict) -> None:
        srtt = obs.get("srtt_ms")
        min_rtt = obs.get("min_rtt_ms")
        if srtt is None or min_rtt is None or min_rtt <= 0:
            return
        self.in_slow_start = False
        qd = max(0.0, srtt - min_rtt)
        # ~1 packet per RTT, pro-rated to the tick length
        step = 5.0 / max(srtt, 5.0)
        if qd < ALPHA_MS:
            flow.cwnd += step
        elif qd > BETA_MS:
            flow.cwnd = max(2.0, flow.cwnd - step)

    def on_rto(self, flow) -> None:
        flow.cwnd = 1.0
        self.in_slow_start = True
