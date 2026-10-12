"""Metric computation from a RunResult.

Definitions (all over the measurement window [warmup_s, duration_s] unless
stated; warmup default 2 s):

- goodput_mbps: unique in-order payload bytes delivered at receivers
  (long flows only) divided by the window length. Retransmitted payload is
  never counted twice. Mbps = megabits/s.
- throughput_mbps: all payload bytes physically delivered at the receiver
  including retransmitted duplicates / window (long flows only).
- mean/median/p95 RTT: sender-side RTT samples (long flows, window).
- mean/p95 queueing delay: RTT sample minus the flow's minimum observed RTT.
- loss_rate: (random + overflow packet losses at the bottleneck) / (all
  packets that arrived at the bottleneck: data + retransmissions), all flows.
- retx_rate: retransmitted packets / data packets sent (long flows).
- jain_fairness: Jain index over long-flow goodput shares in the window.
- goodput_cv: coefficient of variation of per-flow goodput.
- cwnd_osc: mean over long flows of std(cwnd)/mean(cwnd) on 100 ms buckets.
- queue_occ_mean/max_pkts: bottleneck queue occupancy (pk).
- rate_stability: std/mean of aggregate goodput on 100 ms buckets.
- recovery_time_s: per network event (see _recovery_time): seconds until the
  controller re-stabilizes after an abrupt change.
- fct_mean/median/p95_s: flow completion times of short flows.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..environment.sim import RunResult


def _window(result: RunResult):
    sc = result.scenario
    t0 = float(sc["warmup_s"]) * 1000.0
    t1 = float(sc["duration_s"]) * 1000.0
    return t0, t1


def jain(values) -> float:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan")
    if v.sum() == 0:
        return 0.0
    return float(v.sum() ** 2 / (v.size * np.square(v).sum()))


def _bucket_slice(result: RunResult, t0: float, t1: float) -> Optional[slice]:
    bt = np.asarray(result.buckets_t)
    if bt.size == 0:
        return None
    lo = int(np.searchsorted(bt, t0 - 1e-9))
    hi = int(np.searchsorted(bt, t1 + 1e-9, side="right"))
    if hi <= lo:
        hi = min(len(bt), lo + 1)
    return slice(lo, max(hi, lo + 1))


def compute_metrics(result: RunResult) -> Dict:
    sc = result.scenario
    t0, t1 = _window(result)
    window_s = (t1 - t0) / 1000.0
    long_idx = [i for i, pf in enumerate(result.per_flow) if not pf["is_short"]]
    out: Dict = {}

    # ---- per-flow goodput over the window (from 100 ms buckets) -----------
    flow_goodput_bps: List[float] = []
    for i in long_idx:
        pf_buckets = result.bucket_per_flow[i]["goodput_bytes"]
        sl = _bucket_slice(result, t0, t1)
        gb = float(np.sum(pf_buckets[sl])) if sl is not None else 0.0
        flow_goodput_bps.append(gb * 8.0 / window_s / 1e6)
    out["flow_goodput_mbps"] = flow_goodput_bps
    out["goodput_mbps"] = float(sum(flow_goodput_bps))
    out["jain_fairness"] = jain(flow_goodput_bps)
    arr = np.array(flow_goodput_bps, dtype=float)
    out["goodput_cv"] = float(arr.std() / arr.mean()) if arr.size and arr.mean() > 0 else 0.0

    # ---- throughput (all delivered payload, incl. retx duplicates) ------
    thr_bytes = 0.0
    sl = _bucket_slice(result, t0, t1)
    if sl is not None:
        for i in long_idx:
            # per-flow throughput = delivered_all - inorder? Simpler: use
            # flow-level rcv_all_pkts deltas; approximate via totals here:
            pass
        thr_bytes = float(np.sum(result.bucket_agg["throughput_bytes"][sl]))
    out["throughput_mbps"] = thr_bytes * 8.0 / window_s / 1e6

    # ---- RTT / queueing delay samples ------------------------------------
    rtt_all: List[float] = []
    qd_all: List[float] = []
    for i in long_idx:
        pf = result.per_flow[i]
        min_rtt = pf["min_rtt_ms"]
        for t_ms, r in pf["rtt_samples"]:
            if t0 <= t_ms <= t1:
                rtt_all.append(r)
                if min_rtt is not None:
                    qd_all.append(max(0.0, r - min_rtt))
    if rtt_all:
        out["mean_rtt_ms"] = float(np.mean(rtt_all))
        out["median_rtt_ms"] = float(np.median(rtt_all))
        out["p95_rtt_ms"] = float(np.percentile(rtt_all, 95))
        out["n_rtt_samples"] = len(rtt_all)
    else:
        out["mean_rtt_ms"] = out["median_rtt_ms"] = out["p95_rtt_ms"] = None
        out["n_rtt_samples"] = 0
    if qd_all:
        out["mean_qd_ms"] = float(np.mean(qd_all))
        out["p95_qd_ms"] = float(np.percentile(qd_all, 95))
    else:
        out["mean_qd_ms"] = out["p95_qd_ms"] = None

    # ---- loss / retransmissions ------------------------------------------
    lost = result.totals.get("lost_random_pkts", 0) + result.totals.get("lost_overflow_pkts", 0)
    sent_total = sum(pf["sent_data_pkts"] + pf["retx_pkts"] for pf in result.per_flow)
    retx = sum(result.per_flow[i]["retx_pkts"] for i in long_idx)
    data = sum(result.per_flow[i]["sent_data_pkts"] for i in long_idx)
    out["lost_pkts"] = float(lost)
    out["sent_pkts"] = float(sent_total)
    out["loss_rate"] = float(lost / sent_total) if sent_total > 0 else 0.0
    out["retx_rate"] = float(retx / data) if data > 0 else 0.0
    out["rto_events"] = int(result.totals.get("rto_events", 0))
    out["fast_retx_events"] = int(result.totals.get("fast_retx_events", 0))
    out["partial_retx_events"] = int(result.totals.get("partial_retx_events", 0))

    # ---- stability --------------------------------------------------------
    oscs: List[float] = []
    for i in long_idx:
        cw = result.bucket_per_flow[i]["cwnd_mean"]
        sl = _bucket_slice(result, t0, t1)
        if sl is not None:
            c = cw[sl]
            if c.mean() > 0:
                oscs.append(float(c.std() / c.mean()))
    out["cwnd_osc"] = float(np.mean(oscs)) if oscs else None
    if sl is not None:
        g = result.bucket_agg["goodput_bytes"][sl] * 8.0 / 0.1 / 1e6  # per-bucket Mbps
        out["rate_stability"] = float(g.std() / g.mean()) if g.mean() > 0 else None
        q = result.bucket_agg["queue_pkts_mean"][sl]
        out["queue_occ_mean_pkts"] = float(q.mean())
        out["queue_occ_p95_pkts"] = float(np.percentile(q, 95))

    # ---- recovery after abrupt events -------------------------------------
    events = []
    events += [{"t_s": e["t_s"], "type": "cap", "v": e["bandwidth_mbps"]}
               for e in sc.get("capacity_events") or []]
    events += [{"t_s": e["t_s"] / 1000.0, "type": "rtt", "v": e["base_rtt_ms"]}
               for e in sc.get("rtt_events") or []]
    events += [{"t_s": e["t_s"] / 1000.0, "type": "loss", "v": e["loss_rate"]}
               for e in sc.get("loss_events") or []]
    events.sort(key=lambda e: e["t_s"])
    out["recovery_times_s"] = [_recovery_time(result, ev) for ev in events]

    # ---- short flows -------------------------------------------------------
    fcts = result.short_flow_fcts_ms
    if fcts:
        f = np.asarray(fcts, dtype=float) / 1000.0
        out["fct_mean_s"] = float(f.mean())
        out["fct_median_s"] = float(np.median(f))
        out["fct_p95_s"] = float(np.percentile(f, 95))
        out["n_short_flows"] = len(fcts)
    else:
        out["fct_mean_s"] = out["fct_median_s"] = out["fct_p95_s"] = None
        out["n_short_flows"] = 0

    # sanity guard: no NaN/Inf in the primary outputs
    for k in ("goodput_mbps", "mean_rtt_ms", "loss_rate"):
        v = out.get(k)
        if v is not None and not math.isfinite(v):
            out[k] = None
    return out


def _recovery_time(result: RunResult, ev: Dict) -> Optional[float]:
    """Seconds from an abrupt event until the controller re-stabilizes.

    Pre-registered definitions:
    - capacity decrease: time until mean queueing delay falls below 20 ms
      (i.e., the overshoot queue has drained);
    - capacity increase / loss spike / RTT change: time until trailing-1 s
      goodput reaches >= 90% of the bottleneck capacity at that time.
    """
    sc = result.scenario
    t_e = ev["t_s"] * 1000.0
    dur = float(sc["duration_s"]) * 1000.0
    bt = np.asarray(result.buckets_t, dtype=float)
    if bt.size == 0:
        return None
    bsize = (bt[1] - bt[0]) if bt.size > 1 else 100.0

    def cap_at(t_ms: float) -> float:
        bw = float(sc["bandwidth_mbps"])
        for e in sc.get("capacity_events") or []:
            if t_ms >= e["t_s"] * 1000.0:
                bw = e["bandwidth_mbps"]
        if (sc.get("bw_variation") or {}).get("type") == "sine":
            amp = float(sc["bw_variation"].get("amplitude_frac", 0.3))
            per = float(sc["bw_variation"].get("period_s", 10.0)) * 1000.0
            bw *= 1.0 + amp * math.sin(2 * math.pi * t_ms / per)
        return max(bw, 0.05 * float(sc["bandwidth_mbps"]))

    is_decrease = ev["type"] == "cap" and ev["v"] < cap_at(t_e - 1.0)
    if ev["type"] == "cap" and is_decrease:
        qd = result.bucket_agg["qd_mean"] if "qd_mean" in result.bucket_agg else None
        # qd per flow is in bucket_per_flow; aggregate mean across long flows
        long_idx = [i for i, pf in enumerate(result.per_flow) if not pf["is_short"]]
        qd_series = np.zeros(len(bt))
        n = 0
        for i in long_idx:
            qd_series += result.bucket_per_flow[i]["qd_mean"]
            n += 1
        if n:
            qd_series /= n
        for i, t in enumerate(bt):
            if t >= t_e and qd_series[i] < 20.0:
                return (t - t_e) / 1000.0
        return (dur - t_e) / 1000.0

    # goodput-based: trailing 1 s window mean >= 0.9 * capacity at window end
    gb = result.bucket_agg["goodput_bytes"] * 8.0 / (bsize / 1000.0) / 1e6  # Mbps per bucket
    per_sec = int(round(1000.0 / bsize))
    for i in range(len(bt)):
        t = bt[i]
        if t < t_e + 1000.0:
            continue
        lo = max(0, i - per_sec + 1)
        window_mean = float(gb[lo:i + 1].mean())
        if window_mean >= 0.9 * cap_at(t):
            return (t - t_e) / 1000.0
    return (dur - t_e) / 1000.0
