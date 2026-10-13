#!/usr/bin/env python3
"""Trace the Reno recovery crawl at bw20_rtt60: sender/receiver state."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from src.environment.sim import Simulator
from src.controllers.reno import RenoController


class DebugReno(RenoController):
    name = "debug_reno"
    _t = 0.0

    def on_fast_retransmit(self, flow):
        print(f"  FRXT t={flow and self._t:7.1f} cwnd={flow.cwnd:6.1f} una={flow.snd_una} nxt={flow.next_seq}")
        super().on_fast_retransmit(flow)

    def on_tick(self, flow, obs):
        self._t = obs["t_ms"]
        super().on_tick(flow, obs)


def main():
    sc = {'name': 'd', 'duration_s': 4, 'bandwidth_mbps': 20.0, 'base_rtt_ms': 60.0,
          'queue_bdp_factor': 2.0, 'loss_rate': 0.0, 'n_flows': 1, 'warmup_s': 2.0, 'seed': 0}
    sim = Simulator(sc, lambda flow_id=0: DebugReno(flow_id))

    # monkeypatch _sender_on_ack to log state every ~100ms
    orig = sim._sender_on_ack
    last = {"t": 0.0}

    def patched(f, cum, dup, rtt, t, sack):
        orig(f, cum, dup, rtt, t, sack)
        if t - last["t"] >= 100.0 and (t > 900 or t < 500):
            last["t"] = t
            buf = f.rcv_buffer
            print(f"t={t:7.1f} una={f.snd_una:6d} nxt={f.next_seq:6d} cwnd={f.cwnd:7.1f} "
                  f"rcv_nxt={f.rcv_nxt:6d} nbuf={len(buf):3d} buf[:4]={[(s, e) for s, e in buf[:4]]} "
                  f"retxpend={len(f.retx_pending)} recovery={f.recovery_active}")

    sim._sender_on_ack = patched
    res = sim.run()
    print(f"\nevents: rto={sim.rto_events} frxt={sim.fast_retx_events} partial={sim.partial_retx_events}")
    pf = res.per_flow[0]
    print(f"sent={pf['sent_data_pkts']:.0f} retx={pf['retx_pkts']:.0f} inorder={pf['rcv_inorder_pkts']:.0f}")


if __name__ == "__main__":
    main()
