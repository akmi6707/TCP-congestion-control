#!/usr/bin/env python3
"""Temporary diagnostic for simulator physics (kept for regression debugging)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from src.environment.sim import simulate
from src.controllers import make_controller_factory
from src.metrics import compute_metrics


def diag_single():
    base = {
        "name": "diag", "duration_s": 20, "bandwidth_mbps": 20.0,
        "base_rtt_ms": 50.0, "queue_bdp_factor": 2.0, "loss_rate": 0.0,
        "n_flows": 1, "warmup_s": 2.0, "seed": 0,
    }
    res = simulate(base, make_controller_factory("reno"))
    pf = res.per_flow[0]
    sent = pf["sent_data_pkts"] + pf["retx_pkts"]
    delivered = pf["rcv_all_pkts"]
    lost = res.totals["lost_random_pkts"] + res.totals["lost_overflow_pkts"]
    print(f"sent={sent:.0f} delivered={delivered:.0f} lost={lost:.0f} "
          f"rcv_inorder={pf['rcv_inorder_pkts']:.0f} min_rtt={pf['min_rtt_ms']:.1f}")
    print(f"unaccounted (in flight at end): {sent - delivered - lost:.0f}")
    gp = res.bucket_agg["goodput_bytes"] * 8 / 0.1 / 1e6
    q = res.bucket_agg["queue_pkts_mean"]
    cw = res.bucket_per_flow[0]["cwnd_mean"]
    print("t_s  goodput_mbps  queue_pkts  cwnd")
    for i in range(0, len(gp), 10):
        print(f"{res.buckets_t[i]/1000:5.1f} {gp[i]:8.2f} {q[i]:10.1f} {cw[i]:7.1f}")
    m = compute_metrics(res)
    print("metrics:", {k: m[k] for k in ("goodput_mbps", "mean_rtt_ms", "loss_rate", "retx_rate")})


def diag_four():
    sc = {
        "name": "diag4", "duration_s": 40, "bandwidth_mbps": 20.0,
        "base_rtt_ms": 50.0, "queue_bdp_factor": 2.0, "loss_rate": 0.0,
        "n_flows": 4, "warmup_s": 2.0, "seed": 0,
    }
    res = simulate(sc, make_controller_factory("reno"))
    for i, pf in enumerate(res.per_flow):
        print(f"flow{i}: sent={pf['sent_data_pkts']:.0f} retx={pf['retx_pkts']:.0f} "
              f"inorder={pf['rcv_inorder_pkts']:.0f} min_rtt={pf['min_rtt_ms']:.1f} srtt_end={pf['srtt_final_ms']}")
    gp = res.bucket_agg["goodput_bytes"] * 8 / 0.1 / 1e6
    print("aggregate goodput per 1 s:", " ".join(f"{v:.1f}" for v in gp[::10]))
    for i in range(4):
        cwf = res.bucket_per_flow[i]["cwnd_mean"]
        print(f"flow{i} cwnd @1s intervals:", " ".join(f"{v:.0f}" for v in cwf[::10]))


if __name__ == "__main__":
    diag_single()
    print("==== 4-flow ====")
    diag_four()
