#!/usr/bin/env python3
"""Trace BBR-like internal state to find the overflow driver."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from src.environment.sim import Simulator
from src.controllers.bbr_like import BBRLikeController


class DebugBBR(BBRLikeController):
    name = "debug_bbr"
    last_print = 0.0

    def on_tick(self, flow, obs):
        t = obs["t_ms"]
        if t - self.last_print >= 500:
            self.last_print = t
            print(f"t={t:7.0f} phase={self.phase:9s} gain_idx={self.gain_idx} "
                  f"est={obs.get('est_bw_mbps')} bw_max={self.bw_max_mbps:6.2f} "
                  f"min_rtt={obs.get('min_rtt_ms')} cwnd={flow.cwnd:7.1f} "
                  f"inflight={flow.inflight:5d} una={flow.snd_una} nxt={flow.next_seq}")
        super().on_tick(flow, obs)


def main():
    base = {
        "name": "diag", "duration_s": 12, "bandwidth_mbps": 20.0,
        "base_rtt_ms": 50.0, "queue_bdp_factor": 2.0, "loss_rate": 0.0,
        "n_flows": 1, "warmup_s": 2.0, "seed": 0,
    }
    sim = Simulator(base, lambda flow_id: DebugBBR(flow_id=flow_id))
    res = sim.run()
    pf = res.per_flow[0]
    print(f"\nsent={pf['sent_data_pkts']:.0f} retx={pf['retx_pkts']:.0f} "
          f"lost={res.totals['lost_random_pkts'] + res.totals['lost_overflow_pkts']:.0f} "
          f"rto_events={sim.rto_events} frxt={sim.fast_retx_events} partial={sim.partial_retx_events}")


if __name__ == "__main__":
    main()
