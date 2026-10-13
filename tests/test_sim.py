"""Unit tests for simulator invariants: conservation, determinism, bounds."""
import numpy as np
import pytest

from src.environment.sim import simulate
from src.controllers import make_controller_factory


def test_packet_conservation_no_loss(tiny_scenario):
    res = simulate(tiny_scenario, make_controller_factory("reno"))
    pf = res.per_flow[0]
    delivered = pf["rcv_all_pkts"]
    lost = res.totals["lost_random_pkts"] + res.totals["lost_overflow_pkts"]
    sent = pf["sent_data_pkts"] + pf["retx_pkts"]
    # everything sent is delivered, lost, or still in flight at the end
    assert delivered + lost <= sent + 1e-9
    assert sent - delivered - lost >= 0


def test_determinism_same_seed(tiny_scenario):
    r1 = simulate(tiny_scenario, make_controller_factory("reno"))
    r2 = simulate(tiny_scenario, make_controller_factory("reno"))
    assert r1.per_flow[0]["rcv_inorder_pkts"] == r2.per_flow[0]["rcv_inorder_pkts"]
    assert r1.totals == r2.totals


def test_random_loss_reproducible_across_controllers(lossy_scenario):
    # same seed => same noise draw sequence (u arrays are seed-derived)
    r1 = simulate(lossy_scenario, make_controller_factory("reno"))
    r2 = simulate(lossy_scenario, make_controller_factory("reno"))
    assert r1.totals["lost_random_pkts"] == r2.totals["lost_random_pkts"]


def test_loss_actually_happens(lossy_scenario):
    res = simulate(lossy_scenario, make_controller_factory("reno"))
    assert res.totals["lost_random_pkts"] > 0


def test_cwnd_never_negative_or_nan(lossy_scenario):
    sim_state = {"min": 1e18, "max": -1e18, "nan": False}

    class Probe:
        def __init__(self, inner):
            self.inner = inner
            self.name = "probe_" + inner.name

        def on_new_acks(self, f, n):
            self.inner.on_new_acks(f, n)

        def on_fast_retransmit(self, f):
            self.inner.on_fast_retransmit(f)

        def on_rto(self, f):
            self.inner.on_rto(f)

        def on_tick(self, f, obs):
            self.inner.on_tick(f, obs)
            c = f.cwnd
            sim_state["min"] = min(sim_state["min"], c)
            sim_state["max"] = max(sim_state["max"], c)
            if c != c:
                sim_state["nan"] = True

    simulate(lossy_scenario, lambda flow_id=0: Probe(make_controller_factory("reno")(flow_id)))
    assert not sim_state["nan"]
    assert sim_state["min"] >= 1.0
    assert np.isfinite(sim_state["max"])


def test_multi_flow_runs(tiny_scenario):
    sc = dict(tiny_scenario, n_flows=3)
    res = simulate(sc, make_controller_factory("reno"))
    assert len(res.per_flow) == 3
    for pf in res.per_flow:
        assert pf["rcv_inorder_pkts"] > 0


def test_short_flow_completion(tiny_scenario):
    # With a very shallow buffer, a hogging Reno flow can starve short flows
    # (realistic bufferbloat behavior); use a moderate queue so completion
    # is achievable and assert that short flows finish and report FCTs.
    sc = dict(tiny_scenario, duration_s=3.0, queue_bdp_factor=3.0,
              short_flows={"n": 3, "size_pkts_range": [10, 30],
                           "start_s_range": [0.2, 1.0]})
    res = simulate(sc, make_controller_factory("reno"))
    assert len(res.short_flow_fcts_ms) >= 2
    for fct in res.short_flow_fcts_ms:
        assert fct > 0


def test_capacity_event_changes_capacity(tiny_scenario):
    sc = dict(tiny_scenario, duration_s=3.0,
              capacity_events=[{"t_s": 2.0, "bandwidth_mbps": 40.0}])
    from src.environment.sim import Simulator
    sim = Simulator(sc, make_controller_factory("reno"))
    assert sim.capacity_mbps(1000.0) == 4.0
    assert sim.capacity_mbps(2500.0) == 40.0
