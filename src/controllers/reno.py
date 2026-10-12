"""TCP Reno: slow start + loss-based AIMD (RFC 5681-style).

Packet units. Slow start exits on the first loss (or when cwnd reaches
ssthresh). Congestion avoidance grows cwnd by n/cwnd packets per n acks
(+1 packet per RTT). Fast retransmit: ssthresh = cwnd/2, cwnd = ssthresh.
RTO: ssthresh = cwnd/2, cwnd = 1 (then slow start).
"""
from __future__ import annotations

import math

from .base import BaseController


class RenoController(BaseController):
    name = "reno"

    def on_new_acks(self, flow, n_new: int) -> None:
        c = flow.cwnd
        if c < flow.ssthresh:
            flow.cwnd = c + n_new                       # slow start
        else:
            flow.cwnd = c + n_new / max(c, 1.0)          # congestion avoidance

    def on_fast_retransmit(self, flow) -> None:
        ss = max(flow.cwnd / 2.0, 2.0)
        flow.ssthresh = ss
        flow.cwnd = ss

    def on_rto(self, flow) -> None:
        flow.ssthresh = max(flow.cwnd / 2.0, 2.0)
        flow.cwnd = 1.0
