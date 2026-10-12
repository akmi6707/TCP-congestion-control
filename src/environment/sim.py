"""Discrete-time packet-batch TCP dumbbell simulator.

Topology: N senders -> shared bottleneck router (drop-tail FIFO queue) ->
N receivers, one greedy flow per sender/receiver pair (plus optional
short-lived flows). The reverse (ACK) path is assumed congestion-free,
as is standard in dumbbell congestion-control studies.

Model summary (a documented abstraction, not a kernel TCP):
- Time advances in fixed ticks of ``dt`` (default 5 ms). Each flow may send
  ``floor(cwnd) - inflight`` new data packets per tick (ACK-clocked, no
  pacing). Retransmissions are not counted against the window, matching
  loss-recovery behavior.
- Packets sent at tick i arrive at the bottleneck queue at tick i + f_steps,
  where f_steps = base_rtt/(2*dt) (forward propagation delay). Loss at the
  queue happens on arrival: per-packet random loss with probability p(t)
  (deterministic given a shared per-tick uniform draw; see utils.rng.loss_count)
  plus drop-tail overflow, attributed proportionally to arriving batches.
- Within a batch, lost packets are modeled as one contiguous hole at a
  position derived from a second shared per-tick uniform draw.
- Queue service is fluid at rate C(t) (pkts/ms); the queue holds q <= qmax
  packets. A batch's mean queue wait is (packets_ahead + d/2)/rate, which
  preserves FIFO delivery order within a flow.
- Delivered packets reach the receiver at arrival+wait; the receiver advances
  its cumulative ACK point over in-order data and buffers out-of-order
  segments; each delivery generates one ACK event carrying the new cumulative
  point and a duplicate-ACK count for out-of-order packets.
- ACKs return to the sender after the backward propagation delay. The sender
  keeps RFC 6298 SRTT/RTTVAR/RTO state (min RTO 200 ms, max 3 s), performs
  fast retransmit on 3 duplicate ACKs, and NewReno-style retransmission of
  the next hole on each partial ACK during recovery. Retransmitted segments
  are RTT-sampled from the retransmission time (timestamps semantics).
- The controller ONLY chooses cwnd (plus its own ssthresh bookkeeping).
  All loss-recovery mechanics are identical for every controller, so
  comparisons isolate the congestion-control policy.

Determinism: given the scenario config and seed, the per-tick link-noise
uniform draws are fixed, so runs are exactly reproducible. Different
controllers in one scenario see the same noise sequence at the same times;
exact packet-level pairing is impossible because sending patterns differ
(documented limitation).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils import config as cfgmod
from ..utils.rng import loss_count, make_rng


class Flow:
    """Per-flow sender + receiver state."""

    __slots__ = (
        "flow_id", "bidx", "is_short", "start_ms", "total_pkts", "remaining_pkts",
        "cwnd", "ssthresh", "snd_una", "next_seq", "srtt", "rttvar", "rto",
        "min_rtt", "dupacks_pending", "retx_pending", "recovery_active",
        "recovery_high", "rto_base_ms", "controller", "done_sending", "fct_ms",
        "last_tick_new_acks", "last_tick_dupacks", "last_tick_loss_event",
        "last_tick_rto_event",
        "rcv_nxt", "rcv_buffer", "rcv_inorder_pkts", "rcv_all_pkts",
        "rtt_samples", "send_times", "retx_send_times", "last_acked_cum",
        "delivered_log", "delivered_window_bytes", "first_delivered_ms",
        "sent_data", "retx_sent",
    )

    def __init__(self, flow_id: int, start_ms: float, total_pkts: float,
                 is_short: bool, controller) -> None:
        self.flow_id = flow_id
        self.bidx = -1
        self.is_short = is_short
        self.start_ms = start_ms
        self.total_pkts = total_pkts        # math.inf for long flows
        self.remaining_pkts = total_pkts
        # ---- sender
        self.cwnd = 10.0                    # IW10, packets
        self.ssthresh = math.inf
        self.snd_una = 0
        self.next_seq = 0
        self.srtt = None                    # ms
        self.rttvar = None
        self.rto = 200.0                    # ms, initial
        self.min_rtt = math.inf             # ms
        self.dupacks_pending = 0
        self.retx_pending = set()           # seqs retransmitted, not yet acked
        self.retx_send_times: Dict[int, float] = {}
        self.recovery_active = False
        self.recovery_high = -1
        self.rto_base_ms = None
        self.controller = controller
        self.done_sending = False
        self.fct_ms = None
        # ---- per-tick observation scratch (reset at the start of each tick)
        self.last_tick_new_acks = 0
        self.last_tick_dupacks = 0
        self.last_tick_loss_event = False
        self.last_tick_rto_event = False
        # ---- receiver
        self.rcv_nxt = 0
        self.rcv_buffer: List[List[int]] = []   # sorted [start, end) seq ranges
        self.rcv_inorder_pkts = 0
        self.rcv_all_pkts = 0
        self.rtt_samples: List[Tuple[float, float]] = []  # (t_ms, rtt_ms)
        self.last_acked_cum = 0   # cum seq carried by the latest scheduled ACK
        # ---- sender accounting
        self.send_times: List[float] = []
        self.delivered_log: deque = deque()  # (t_ms, bytes)
        self.delivered_window_bytes = 0.0
        self.first_delivered_ms = None
        self.sent_data = 0.0
        self.retx_sent = 0.0

    @property
    def inflight(self) -> int:
        return self.next_seq - self.snd_una

    def est_bw_bytes_per_ms(self, now_ms: float, window_ms: float = 1000.0) -> float:
        log = self.delivered_log
        while log and now_ms - log[0][0] > window_ms:
            _, b = log.popleft()
            self.delivered_window_bytes -= b
        # Clamp the effective averaging window to [100 ms, window]: early in
        # a flow's life, dividing by only the few elapsed ms turns any large
        # ACK-drain chunk into a huge instantaneous burst-rate estimate
        # (Phase-2 bug: BBR-like read 116 Mbps on a 20 Mbps link).
        eff_window = window_ms
        if self.first_delivered_ms is not None:
            eff_window = min(window_ms, max(now_ms - self.first_delivered_ms, 100.0))
        return self.delivered_window_bytes / eff_window

    def update_rtt(self, rtt_ms: float, t_ms: float) -> None:
        self.rtt_samples.append((t_ms, rtt_ms))
        if rtt_ms < self.min_rtt:
            self.min_rtt = rtt_ms
        if self.srtt is None:
            self.srtt = rtt_ms
            self.rttvar = rtt_ms / 2.0
        else:
            self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - rtt_ms)
            self.srtt = 0.875 * self.srtt + 0.125 * rtt_ms
        self.rto = min(3000.0, max(200.0, self.srtt + max(10.0, 4.0 * self.rttvar)))

    # ---- receiver ---------------------------------------------------------
    def insert_buffer(self, s: int, e: int) -> None:
        buf = self.rcv_buffer
        idx = 0
        while idx < len(buf) and buf[idx][0] < s:
            idx += 1
        if idx > 0 and buf[idx - 1][1] >= s:
            buf[idx - 1][1] = max(buf[idx - 1][1], e)
            j = idx
            while j < len(buf) and buf[j][0] <= buf[idx - 1][1]:
                buf[idx - 1][1] = max(buf[idx - 1][1], buf[j][1])
                j += 1
            del buf[idx:j]
        else:
            buf.insert(idx, [s, e])
            j = idx + 1
            while j < len(buf) and buf[j][0] <= e:
                buf[idx][1] = max(buf[idx][1], buf[j][1])
                buf.pop(j)


@dataclass
class RunResult:
    """Everything needed to compute metrics for one run."""
    scenario: Dict
    per_flow: List[Dict] = field(default_factory=list)     # exact per-flow data
    totals: Dict = field(default_factory=dict)
    buckets_t: List[float] = field(default_factory=list)
    bucket_agg: Dict[str, np.ndarray] = field(default_factory=dict)
    bucket_per_flow: List[Dict[str, np.ndarray]] = field(default_factory=list)
    short_flow_fcts_ms: List[float] = field(default_factory=list)
    controller_info: Dict = field(default_factory=dict)


class Simulator:
    """Dumbbell bottleneck simulation."""

    def __init__(self, scenario: Dict, controller_factory, seed: Optional[int] = None,
                 series_bucket_ms: float = 100.0) -> None:
        self.sc = cfgmod.validate_scenario(dict(scenario))
        if seed is not None:
            self.sc["seed"] = seed
        self.seed = int(self.sc["seed"])
        self.dt = float(self.sc["dt_ms"])
        self.duration_ms = float(self.sc["duration_s"]) * 1000.0
        self.n_ticks = int(round(self.duration_ms / self.dt))
        self.mss = cfgmod.MSS_BYTES
        self.payload = cfgmod.PAYLOAD_BYTES

        self.base_bw = float(self.sc["bandwidth_mbps"])
        self.capacity_events = sorted(self.sc["capacity_events"], key=lambda e: e["t_s"])
        self.bw_var = self.sc["bw_variation"]

        base_rtt = float(self.sc["base_rtt_ms"])
        c0 = self.pkts_per_ms(self.base_bw)
        if self.sc["queue_packets"] is not None:
            self.qmax = float(self.sc["queue_packets"])
        else:
            self.qmax = float(self.sc["queue_bdp_factor"]) * c0 * base_rtt

        self.rtt_events = [(e["t_s"] * 1000.0, e["base_rtt_ms"]) for e in self.sc["rtt_events"]]
        self.loss_events = [(e["t_s"] * 1000.0, e["loss_rate"]) for e in self.sc["loss_events"]]

        rng = make_rng(self.seed, "link_noise")
        self.u = rng.random((self.n_ticks + 8, 2))  # per-tick [loss, hole] draws

        n_long = int(self.sc["n_flows"])
        offsets = self.sc["flow_start_offsets_s"] or [0.0] * n_long
        self.flows: List[Flow] = []
        self.short_flows: List[Flow] = []
        self.controller_factory = controller_factory
        for i in range(n_long):
            ctrl = controller_factory(flow_id=i)
            f = Flow(i, offsets[i] * 1000.0, math.inf, False, ctrl)
            f.bidx = len(self.flows)
            self.flows.append(f)
        sf = self.sc["short_flows"]
        if sf and sf["n"] > 0:
            srng = make_rng(self.seed, "short_flows")
            for j in range(sf["n"]):
                size = int(srng.integers(sf["size_pkts_range"][0], sf["size_pkts_range"][1] + 1))
                start = float(srng.uniform(sf["start_s_range"][0], sf["start_s_range"][1]))
                ctrl = controller_factory(flow_id=1000 + j)
                f = Flow(1000 + j, start * 1000.0, size, True, ctrl)
                f.bidx = len(self.flows) + len(self.short_flows)
                self.short_flows.append(f)

        self.arrivals: Dict[int, List[Tuple[Flow, int, int, bool]]] = {}
        self.ack_events: Dict[int, List[Tuple[Flow, int, int, float]]] = {}

        self.q = 0.0
        self.lost_random = 0.0
        self.lost_overflow = 0.0
        self.rto_events = 0
        self.fast_retx_events = 0
        self.partial_retx_events = 0

        self.series_bucket_ms = series_bucket_ms
        self.bucket_len = max(1, int(round(series_bucket_ms / self.dt)))
        n_buckets = self.n_ticks // self.bucket_len + 1
        self.bucket = {
            "goodput_bytes": np.zeros(n_buckets),
            "throughput_bytes": np.zeros(n_buckets),
            "queue_pkts_sum": np.zeros(n_buckets),
            "queue_pkts_max": np.zeros(n_buckets),
            "loss_pkts": np.zeros(n_buckets),
            "cap_mbps_sum": np.zeros(n_buckets),
        }
        keys = ("cwnd_sum", "inflight_sum", "goodput_bytes", "srtt_sum", "qd_sum", "n_valid")
        self.per_flow_bucket = [{k: np.zeros(n_buckets) for k in keys} for _ in self.all_flows()]

    # ------------------------------------------------------------------ util
    def all_flows(self) -> List[Flow]:
        return self.flows + self.short_flows

    def pkts_per_ms(self, mbps: float) -> float:
        return mbps * 1e6 / 8.0 / self.mss / 1000.0

    def capacity_mbps(self, t_ms: float) -> float:
        bw = self.base_bw
        for ev in self.capacity_events:
            if t_ms >= ev["t_s"] * 1000.0:
                bw = ev["bandwidth_mbps"]
        if self.bw_var.get("type") == "sine":
            amp = float(self.bw_var.get("amplitude_frac", 0.3))
            period = float(self.bw_var.get("period_s", 10.0)) * 1000.0
            bw *= 1.0 + amp * math.sin(2 * math.pi * t_ms / period)
        return max(bw, 0.05 * self.base_bw)

    def base_rtt_ms(self, t_ms: float) -> float:
        rtt = float(self.sc["base_rtt_ms"])
        for t_ev, r_ev in self.rtt_events:
            if t_ms >= t_ev:
                rtt = r_ev
        return rtt

    def loss_rate(self, t_ms: float) -> float:
        p = float(self.sc["loss_rate"])
        for t_ev, p_ev in self.loss_events:
            if t_ms >= t_ev:
                p = p_ev
        return p

    def f_steps(self, t_ms: float) -> int:
        return int(round(self.base_rtt_ms(t_ms) / (2 * self.dt)))

    # -------------------------------------------------------------- main loop
    def run(self) -> RunResult:
        for tick in range(self.n_ticks):
            self._tick(tick * self.dt, tick)
        return self._collect()

    def _tick(self, t: float, tick: int) -> None:
        dt = self.dt
        cap_mbps = self.capacity_mbps(t)
        rate_pkts_ms = self.pkts_per_ms(cap_mbps)
        cap_dt = rate_pkts_ms * dt
        p_loss = self.loss_rate(t)

        active = [f for f in self.all_flows() if t >= f.start_ms and not f.done_sending]

        # 1) reset per-tick observation counters (BEFORE ack processing so the
        #    controller's on_tick for this tick sees this tick's events)
        for f in active:
            f.last_tick_new_acks = 0
            f.last_tick_dupacks = 0
            f.last_tick_loss_event = False
            f.last_tick_rto_event = False

        # 2) ACK events due now
        events = self.ack_events.pop(tick, None)
        if events:
            for flow, cum, dup, rtt_sample, sack in events:
                self._sender_on_ack(flow, cum, dup, rtt_sample, t, sack)

        # 3) RTO checks
        for f in active:
            self._rto_check(f, t)

        # 4) controller tick (observation -> policy -> cwnd adjustment)
        for f in active:
            self._controller_tick(f, t, tick)

        # 5) sends
        f_steps_now = self.f_steps(t)
        for f in active:
            self._send(f, t, tick, f_steps_now)

        # 6) bottleneck queue step
        batches = self.arrivals.pop(tick, None)
        if batches or self.q > 0:
            self._queue_step(t, tick, batches or [], rate_pkts_ms, cap_dt, p_loss)

        # 7) series bookkeeping
        b = tick // self.bucket_len
        nb = len(self.bucket["queue_pkts_max"])
        if b < nb:
            agg = self.bucket
            agg["queue_pkts_sum"][b] += self.q
            agg["queue_pkts_max"][b] = max(agg["queue_pkts_max"][b], self.q)
            agg["cap_mbps_sum"][b] += cap_mbps
            for f in active:
                pf = self.per_flow_bucket[f.bidx]
                pf["cwnd_sum"][b] += f.cwnd
                pf["inflight_sum"][b] += f.inflight
                if f.srtt is not None:
                    pf["srtt_sum"][b] += f.srtt
                    if f.min_rtt != math.inf:
                        pf["qd_sum"][b] += max(0.0, f.srtt - f.min_rtt)
                    pf["n_valid"][b] += 1

    # ------------------------------------------------------------ sender logic
    def _sender_on_ack(self, f: Flow, cum: int, dup: int, rtt_sample: float,
                       t: float, sack: tuple) -> None:
        if rtt_sample is not None and rtt_sample > 0:
            f.update_rtt(rtt_sample, t)
        if cum > f.snd_una:
            n_new = cum - f.snd_una
            f.snd_una = cum
            f.dupacks_pending = 0
            if f.retx_pending:
                f.retx_pending = {s for s in f.retx_pending if s >= cum}
                for s in list(f.retx_send_times.keys()):
                    if s < cum:
                        del f.retx_send_times[s]
            bytes_new = n_new * self.payload
            if f.first_delivered_ms is None:
                f.first_delivered_ms = t
            f.delivered_log.append((t, bytes_new))
            f.delivered_window_bytes += bytes_new
            f.last_tick_new_acks += n_new
            if f.recovery_active:
                if f.snd_una >= f.recovery_high:
                    f.recovery_active = False
                else:
                    # SACK/RACK-style recovery: everything below the first
                    # SACK block was sent before delivered data and is
                    # presumed lost; retransmit the missing run (skipping
                    # seqs already retransmitted and unacked). Bounded per
                    # event to avoid retransmission floods. Single-segment
                    # fallback when no SACK info exists (like a plain RTO).
                    seq = f.snd_una
                    sacked = any(s <= seq < e for s, e in sack)
                    if not sacked:
                        run_end = seq + 1
                        if sack:
                            run_end = max(run_end, min(s for s, _ in sack))
                        run_end = min(run_end, seq + 128)
                        n_re = 0
                        for s in range(seq, run_end):
                            if s >= f.next_seq or n_re >= 128:
                                break
                            if s not in f.retx_pending:
                                self._retransmit(f, s, t)
                                n_re += 1
                        self.partial_retx_events += 1
            f.controller.on_new_acks(f, n_new)
        elif dup > 0:
            f.last_tick_dupacks += dup
            if not f.recovery_active and f.snd_una not in f.retx_pending and f.inflight > 0:
                f.dupacks_pending += dup
                if f.dupacks_pending >= 3:
                    f.dupacks_pending = 0
                    f.recovery_active = True
                    f.recovery_high = f.next_seq
                    f.last_tick_loss_event = True
                    self.fast_retx_events += 1
                    f.controller.on_fast_retransmit(f)
                    self._retransmit(f, f.snd_una, t)

    def _retransmit(self, f: Flow, seq: int, t: float) -> None:
        f.retx_sent += 1
        f.retx_pending.add(seq)
        f.retx_send_times[seq] = t
        arrive = int(round(t / self.dt)) + self.f_steps(t)
        self.arrivals.setdefault(arrive, []).append((f, seq, 1, True))

    def _rto_check(self, f: Flow, t: float) -> None:
        # RTO base = time of the last (re)transmission of the oldest
        # outstanding sequence. Computed on demand: with batched ACK-clocking,
        # snd_una often catches up with next_seq each RTT, so a cached value
        # would go stale and fire spuriously (found in Phase-2 debugging).
        if f.inflight <= 0 or f.snd_una >= f.next_seq:
            return
        base = f.retx_send_times.get(f.snd_una)
        if base is None:
            if f.snd_una < len(f.send_times):
                base = f.send_times[f.snd_una]
            else:
                return
        if t - base > f.rto:
            f.rto = min(3000.0, f.rto * 2.0)
            f.last_tick_rto_event = True
            if not f.recovery_active:
                f.recovery_active = True
                f.recovery_high = f.next_seq
            self.rto_events += 1
            f.controller.on_rto(f)
            self._retransmit(f, f.snd_una, t)

    def _send(self, f: Flow, t: float, tick: int, f_steps: int) -> None:
        avail = int(math.floor(f.cwnd)) - f.inflight
        if avail <= 0:
            return
        if f.is_short:
            avail = min(avail, int(f.remaining_pkts))
            if avail <= 0:
                return
        first = f.next_seq
        f.send_times.extend([t] * avail)
        f.next_seq += avail
        f.sent_data += avail
        if f.is_short:
            f.remaining_pkts -= avail
            if f.remaining_pkts <= 0:
                f.done_sending = True
        self.arrivals.setdefault(tick + f_steps, []).append((f, first, avail, False))

    # ---------------------------------------------------------- queue/receiver
    def _queue_step(self, t: float, tick: int, batches, rate_pkts_ms: float,
                    cap_dt: float, p_loss: float) -> None:
        u_loss = self.u[tick, 0] if tick < len(self.u) else 0.5
        u_hole = self.u[tick, 1] if tick < len(self.u) else 0.5
        b_idx = tick // self.bucket_len
        nb = len(self.bucket["loss_pkts"])

        # random loss (shared per-tick draw => same noise sequence for all
        # controllers in a scenario)
        prepared = []  # [flow, first_seq, n, is_retx, k_total, hole_at_tail]
        total_arr = 0.0
        for f, first, n, is_retx in batches:
            k = loss_count(n, p_loss, u_loss)
            self.lost_random += k
            if b_idx < nb:
                self.bucket["loss_pkts"][b_idx] += k
            prepared.append([f, first, n, is_retx, k, False])
            total_arr += n - k

        # Drop-tail overflow: excess arrivals beyond (queue space + service
        # during the tick) are dropped as a CONTIGUOUS TAIL RUN of the arrival
        # stream, like a real FIFO queue. (Proportional per-batch attribution
        # fragments one overflow into hundreds of scattered holes and makes
        # NewReno-style recovery take seconds - found in Phase-2 debugging.)
        # Batches within a tick are simultaneous; the drop order rotates by
        # tick so no flow is systematically the victim.
        space = max(0.0, self.qmax - self.q)
        room = int(math.floor(space + cap_dt))
        overflow = int(max(0, int(total_arr) - room))
        if overflow > 0:
            nbatches = len(prepared)
            start = tick % nbatches
            remaining = overflow
            for j in range(nbatches):
                idx = (start - j) % nbatches
                x = prepared[idx]
                avail_drop = x[2] - x[4]
                take = min(remaining, avail_drop)
                if take > 0:
                    x[4] += take
                    x[5] = True
                remaining -= take
                if remaining <= 0:
                    break
            self.lost_overflow += overflow

        total_eff = int(total_arr) - overflow
        if total_eff < 0:
            total_eff = 0
        q_before = self.q
        served = min(q_before + total_eff, cap_dt)

        pos = 0.0
        for f, first, n, is_retx, k_total, hole_at_tail in prepared:
            d = n - k_total
            if d <= 0:
                continue
            hole_pos = None
            if k_total > 0:
                if hole_at_tail:
                    hole_pos = d          # contiguous tail hole [d, n)
                else:
                    hole_pos = min(int(u_hole * (d + 1)), d)
            wait_ms = (q_before + pos + d / 2.0) / rate_pkts_ms
            cum, dup = self._rcv_process(f, first, d, n, hole_pos, t + wait_ms)
            tau_b = self.base_rtt_ms(t) / 2.0
            if is_retx:
                send_dt = f.retx_send_times.get(first, f.send_times[first] if first < len(f.send_times) else t)
            else:
                send_dt = f.send_times[first]
            rtt_sample = (t + wait_ms + tau_b) - send_dt
            if rtt_sample <= 0:
                rtt_sample = max(tau_b, 1.0)
            # Receiver SACK blocks (delivered out-of-order ranges), for
            # loss recovery: copy because receiver state keeps mutating.
            sack = tuple((s, e) for s, e in f.rcv_buffer[:4])
            # Spread the batch's ACKs across the bottleneck drain window
            # (self-clocking: the receiver ACKs as packets are drained).
            # The spread length is based on max(batch size, cumulative
            # advance): a hole-fill retransmission can drain a large buffered
            # out-of-order region, and that advance must also be spread,
            # otherwise the sender's window reopens in a single tick and
            # dumps a full window at once (found in Phase-2 debugging).
            chunk_pkts = max(1.0, rate_pkts_ms * self.dt)
            last = f.last_acked_cum
            advance = cum - last
            m = max(1, int(math.ceil(max(d, advance) / chunk_pkts)))
            drain_ms = max(d, advance) / rate_pkts_ms
            prev_cum_i = last
            dup_target_prev = 0
            for i in range(1, m + 1):
                cum_i = last + (advance * i) // m
                dup_target = (dup * i) // m
                dup_i = dup_target - dup_target_prev
                dup_target_prev = dup_target
                if cum_i <= prev_cum_i and dup_i <= 0:
                    continue
                ack_t = t + wait_ms + tau_b + drain_ms * (i - 1) / m
                ack_dt = int(round(ack_t / self.dt))
                rtt_here = rtt_sample if i == 1 else None
                self.ack_events.setdefault(ack_dt, []).append((f, cum_i, dup_i, rtt_here, sack))
                prev_cum_i = max(prev_cum_i, cum_i)
            f.last_acked_cum = cum
            pos += d

        self.q = q_before + total_eff - served
        if self.q < 0:
            self.q = 0.0

    def _rcv_process(self, f: Flow, first: int, d: int, n: int,
                     hole_pos: Optional[int], delivery_t: float) -> Tuple[int, int]:
        """Receiver state machine; returns (cum_after, dup_count)."""
        if d <= 0:
            return f.rcv_nxt, 0
        prev_inorder = f.rcv_inorder_pkts
        dup = 0
        if first == f.rcv_nxt:
            if hole_pos is None:
                f.rcv_nxt = first + d
            else:
                f.rcv_nxt = first + hole_pos
                k = n - d
                s = first + hole_pos + k
                e = first + n
                if e > s:
                    f.insert_buffer(s, e)
                dup = d - hole_pos
            while f.rcv_buffer and f.rcv_buffer[0][0] <= f.rcv_nxt:
                s, e = f.rcv_buffer.pop(0)
                if e > f.rcv_nxt:
                    f.rcv_nxt = e
        else:
            if hole_pos is None:
                s, e = first, first + d
            else:
                k = n - d
                s, e = first + hole_pos + k, first + n
            if e > s:
                f.insert_buffer(s, e)
            dup = d

        f.rcv_all_pkts += d
        nb = len(self.bucket["goodput_bytes"])
        b = int(delivery_t // self.series_bucket_ms)
        # every physically delivered packet counts once toward throughput
        if d > 0 and b < nb:
            self.bucket["throughput_bytes"][b] += d * self.payload
        if f.rcv_nxt > prev_inorder:
            advance = f.rcv_nxt - prev_inorder
            f.rcv_inorder_pkts = f.rcv_nxt
            if b < nb:
                self.bucket["goodput_bytes"][b] += advance * self.payload
            if 0 <= f.bidx < len(self.per_flow_bucket):
                pfb = self.per_flow_bucket[f.bidx]["goodput_bytes"]
                if b < len(pfb):
                    pfb[b] += advance * self.payload
            if f.is_short and f.rcv_nxt >= f.total_pkts and f.fct_ms is None:
                f.fct_ms = delivery_t - f.start_ms
        return f.rcv_nxt, dup

    # ------------------------------------------------------------ controller
    def _controller_tick(self, f: Flow, t: float, tick: int) -> None:
        est = None
        if f.delivered_window_bytes > 0:
            # est_bw_bytes_per_ms returns bytes per MILLISECOND:
            # bytes/ms * 1000 = bytes/s; * 8 = bits/s; / 1e6 = Mbps
            est = f.est_bw_bytes_per_ms(t) * 8.0 * 1000.0 / 1e6
        obs = {
            "tick": tick,
            "t_ms": t,
            "cwnd": f.cwnd,
            "inflight": f.inflight,
            "ssthresh": None if f.ssthresh == math.inf else f.ssthresh,
            "srtt_ms": f.srtt,
            "min_rtt_ms": None if f.min_rtt == math.inf else f.min_rtt,
            "est_bw_mbps": est,
            "new_acks": f.last_tick_new_acks,
            "dupacks_pending": f.dupacks_pending,
            "loss_event": f.last_tick_loss_event,
            "rto_event": f.last_tick_rto_event,
            "done_sending": f.done_sending,
            "is_short": f.is_short,
        }
        f.controller.on_tick(f, obs)

    # ---------------------------------------------------------------- collect
    def _collect(self) -> RunResult:
        n_buckets = self.n_ticks // self.bucket_len
        tpb = np.array([min(self.bucket_len, self.n_ticks - b * self.bucket_len)
                        for b in range(n_buckets)], dtype=float)
        tpb = np.maximum(tpb, 1.0)
        rename = {
            "queue_pkts_sum": "queue_pkts_mean",
            "cap_mbps_sum": "cap_mbps_mean",
            "cwnd_sum": "cwnd_mean",
            "inflight_sum": "inflight_mean",
            "srtt_sum": "srtt_mean",
            "qd_sum": "qd_mean",
        }
        bucket = {}
        for k, v in self.bucket.items():
            v = v[:n_buckets]
            if k in rename:
                bucket[rename[k]] = v / tpb
            else:
                bucket[k] = v.copy()
        per_flow_buckets = []
        for pf in self.per_flow_bucket:
            d = {}
            for k, v in pf.items():
                v = v[:n_buckets]
                if k in rename:
                    d[rename[k]] = v / tpb
                elif k == "n_valid":
                    d[k] = v / tpb  # fraction of ticks with valid rtt
                else:
                    d[k] = v.copy()
            per_flow_buckets.append(d)
        per_flow = []
        for f in self.all_flows():
            per_flow.append({
                "flow_id": f.flow_id,
                "is_short": f.is_short,
                "start_ms": f.start_ms,
                "total_pkts": None if f.total_pkts == math.inf else f.total_pkts,
                "sent_data_pkts": f.sent_data,
                "retx_pkts": f.retx_sent,
                "rcv_inorder_pkts": f.rcv_inorder_pkts,
                "rcv_all_pkts": f.rcv_all_pkts,
                "min_rtt_ms": None if f.min_rtt == math.inf else f.min_rtt,
                "srtt_final_ms": f.srtt,
                "rtt_samples": f.rtt_samples,
                "fct_ms": f.fct_ms,
            })
        totals = {
            "lost_random_pkts": self.lost_random,
            "lost_overflow_pkts": self.lost_overflow,
            "rto_events": self.rto_events,
            "fast_retx_events": self.fast_retx_events,
            "partial_retx_events": self.partial_retx_events,
        }
        return RunResult(
            scenario=self.sc,
            per_flow=per_flow,
            totals=totals,
            buckets_t=[b * self.series_bucket_ms for b in range(n_buckets)],
            bucket_agg=bucket,
            bucket_per_flow=per_flow_buckets,
            short_flow_fcts_ms=[f.fct_ms for f in self.short_flows if f.fct_ms is not None],
        )


def simulate(scenario: Dict, controller_factory, seed: Optional[int] = None,
             series_bucket_ms: float = 100.0) -> RunResult:
    sim = Simulator(scenario, controller_factory, seed=seed, series_bucket_ms=series_bucket_ms)
    return sim.run()
