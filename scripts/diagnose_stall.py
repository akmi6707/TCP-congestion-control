#!/usr/bin/env python3
"""Debug stalls + recovery behavior with event counters (kept for regression)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from src.environment.sim import Simulator
from src.controllers.reno import RenoController
from src.metrics import compute_metrics


class DebugReno(RenoController):
    name = "debug_reno"

    def on_rto(self, flow):
        print(f"  RTO  t={self._t:7.1f} rto={flow.rto:7.1f} srtt={flow.srtt} "
              f"cwnd={flow.cwnd:6.1f} inflight={flow.inflight} una={flow.snd_una}")
        super().on_rto(flow)

    def on_fast_retransmit(self, flow):
        print(f"  FRXT t={self._t:7.1f} cwnd={flow.cwnd:6.1f} inflight={flow.inflight} una={flow.snd_una}")
        super().on_fast_retransmit(flow)

    def on_tick(self, flow, obs):
        self._t = obs["t_ms"]
        super().on_tick(flow, obs)


def main():
    base = {
        "name": "diag", "duration_s": 20, "bandwidth_mbps": 20.0,
        "base_rtt_ms": 50.0, "queue_bdp_factor": 2.0, "loss_rate": 0.0,
        "n_flows": 1, "warmup_s": 2.0, "seed": 0,
    }
    sim = Simulator(base, lambda flow_id: DebugReno(flow_id=flow_id))
    res = sim.run()
    pf = res.per_flow[0]
    print(f"\nsent={pf['sent_data_pkts']:.0f} retx={pf['retx_pkts']:.0f} inorder={pf['rcv_inorder_pkts']:.0f}")
    print(f"events: rto={sim.rto_events} frxt={sim.fast_retx_events} partial={sim.partial_retx_events}")
    m = compute_metrics(res)
    print("metrics:", {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in m.items() if not isinstance(v, list)})
    ts = np.array([t for t, r in pf["rtt_samples"]])
    if len(ts) > 1:
        gaps = np.diff(ts)
        print(f"n_ack_events={len(ts)} max_ack_gap_ms={gaps.max():.1f}")
    gp = res.bucket_agg["goodput_bytes"] * 8 / 0.1 / 1e6
    print("goodput/s:", " ".join(f"{v:.1f}" for v in gp[::10]))


if __name__ == "__main__":
    main()
