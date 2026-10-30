
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
        if self.in_slow_start:
            flow.cwnd += n_new

    def on_fast_retransmit(self, flow) -> None:
        # leave slow start on loss; mild backoff then delay control resumes
        self.in_slow_start = False
        flow.cwnd = max(2.0, min(flow.cwnd, flow.ssthresh if flow.ssthresh != float('inf') else flow.cwnd))

    def on_tick(self, flow, obs: dict) -> None:
        srtt = obs.get("srtt_ms")
        min_rtt = obs.get("min_rtt_ms")
        if srtt is None or min_rtt is None or min_rtt <= 0:
            return
        qd = max(0.0, srtt - min_rtt)
        if self.in_slow_start:
            if qd > BETA_MS:
                self.in_slow_start = False
            return
        # ~1 packet per RTT, pro-rated to the tick length
        step = 5.0 / max(srtt, 5.0)
        if qd < ALPHA_MS:
            flow.cwnd += step
        elif qd > BETA_MS:
            flow.cwnd = max(2.0, flow.cwnd - step)

    def on_rto(self, flow) -> None:
        flow.cwnd = 1.0
        self.in_slow_start = True
