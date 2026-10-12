"""BBR-inspired model-based controller (NOT kernel BBR; documented divergence).

Core idea (per BBRv1): maintain (1) a max delivered-bandwidth filter measured
over ~RTT rounds (windowed max over the last 10 rounds, like BBR's BtlBw
filter) and (2) a min-RTT estimate, then set the send window to
pacing_gain x BDP estimate. (A fixed 1 s sliding average lags the startup
ramp and locks the controller at a low fixed point - Phase-2 finding; BBR's
round-based filter does not.)

Phases:
- Startup: gain 2.0 until round-rate growth stalls for 3 consecutive rounds.
- ProbeBW: cycle pacing gains [1.25, 0.75, 1, 1, 1, 1, 1, 1], one min-RTT
  per phase (approximation without packet pacing).
- ProbeRTT: every 10 s, hold cwnd = 4 packets for one min-RTT to re-measure.

Divergences from kernel BBR (documented): no packet-level pacing (window
emulation only), running min-RTT instead of windowed, simplified phase
timing, and a single shared bottleneck. Responds to RTO (transport collapse)
but otherwise ignores loss, like BBR.
"""
from __future__ import annotations

import math

from .base import BaseController

PROBE_BW_GAINS = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
STARTUP_GAIN = 2.0
PROBE_RTT_INTERVAL_MS = 10_000.0
PAYLOAD = 1460.0
MAX_ROUNDS = 10


class BBRLikeController(BaseController):
    name = "bbr_like"

    def __init__(self, flow_id: int = 0) -> None:
        super().__init__(flow_id)
        self.phase = "startup"
        self.gain_idx = 0
        self.phase_until_ms = 0.0
        # round-based delivery-rate filter (packets delivered per round)
        self.round_pkts = 0.0
        self.round_start_ms: float | None = None
        self.round_rates_mbps: list = []   # last MAX_ROUNDS round rates
        self.startup_plateau = 0
        self.probe_rtt_next = PROBE_RTT_INTERVAL_MS
        self.probe_rtt_until = 0.0
        self.last_t = 0.0

    # -------------------------------------------------------------bw filter
    def _close_round_if_due(self, t: float, min_rtt_ms: float | None) -> None:
        if min_rtt_ms is None:
            return
        dur_needed = max(min_rtt_ms, 20.0)
        if self.round_start_ms is None:
            self.round_start_ms = t
            return
        dur = t - self.round_start_ms
        if dur >= dur_needed:
            rate_mbps = self.round_pkts * PAYLOAD * 8.0 / dur / 1000.0
            self.round_rates_mbps.append(rate_mbps)
            if len(self.round_rates_mbps) > MAX_ROUNDS:
                self.round_rates_mbps.pop(0)
            self.round_pkts = 0.0
            self.round_start_ms = t

    @property
    def bw_max_mbps(self) -> float:
        return max(self.round_rates_mbps) if self.round_rates_mbps else 0.0

    def on_new_acks(self, flow, n_new: int) -> None:
        self.round_pkts += n_new

    # -------------------------------------------------------------main tick
    def on_tick(self, flow, obs: dict) -> None:
        t = obs["t_ms"]
        self.last_t = t
        min_rtt = obs.get("min_rtt_ms")
        self._close_round_if_due(t, min_rtt)

        if self.phase == "startup":
            rates = self.round_rates_mbps
            if len(rates) >= 2 and rates[-1] > 0:
                growth = rates[-1] / max(rates[-2], 1e-9)
                self.startup_plateau = self.startup_plateau + 1 if growth < 1.25 else 0
            if self.startup_plateau >= 3 or (t > 5_000 and self.bw_max_mbps > 0):
                self.phase = "probe_bw"
                self.gain_idx = 0
                self.phase_until_ms = t + self._phase_len_ms(min_rtt)
            if self.phase == "startup":
                self._apply(flow, STARTUP_GAIN, min_rtt)
                return

        if self.phase != "probe_rtt" and t >= self.probe_rtt_next and min_rtt is not None:
            self.phase = "probe_rtt"
            self.probe_rtt_until = t + self._phase_len_ms(min_rtt)
        if self.phase == "probe_rtt":
            flow.cwnd = max(4.0, min(flow.cwnd, 4.0))
            if t >= self.probe_rtt_until:
                self.phase = "probe_bw"
                self.gain_idx = 0
                self.phase_until_ms = t + self._phase_len_ms(min_rtt)
                self.probe_rtt_next = t + PROBE_RTT_INTERVAL_MS
            return

        if t >= self.phase_until_ms or min_rtt is None:
            self.gain_idx = (self.gain_idx + 1) % len(PROBE_BW_GAINS)
            self.phase_until_ms = t + self._phase_len_ms(min_rtt)
        gain = PROBE_BW_GAINS[self.gain_idx]
        self._apply(flow, gain, min_rtt)

    def _phase_len_ms(self, min_rtt_ms: float | None) -> float:
        return max(min_rtt_ms or 100.0, 25.0)

    def _apply(self, flow, gain: float, min_rtt_ms: float | None) -> None:
        if min_rtt_ms is None or self.bw_max_mbps <= 0:
            return  # not enough measurements yet; keep current cwnd
        bdp_pkts = self.bw_max_mbps * 1e6 / 8.0 / PAYLOAD * (min_rtt_ms / 1000.0)
        cwnd = gain * bdp_pkts
        cwnd = min(cwnd, 2.5 * bdp_pkts)   # BBR-ish inflight cap
        flow.cwnd = max(4.0, cwnd)

    def on_rto(self, flow) -> None:
        flow.cwnd = 4.0
        self.phase = "startup"
        self.startup_plateau = 0
        self.round_rates_mbps = []
        self.round_pkts = 0.0
        self.round_start_ms = None
