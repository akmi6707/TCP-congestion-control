"""Scenario curriculum sampling for RL training (from configs/curriculum.yaml)."""
from __future__ import annotations

import math
from typing import Dict

from ..utils.rng import make_rng


def _sample_log_uniform(rng, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_scenario(cfg: Dict, episode: int, seed: int) -> Dict:
    """Deterministically sample one training scenario for an episode."""
    rng = make_rng(seed, f"curriculum_ep{episode}")
    s = cfg["sampling"]
    bw = _sample_log_uniform(rng, *s["bandwidth_mbps"]["log_uniform"])
    rtts = s["base_rtt_ms"]["choices"]
    rtt = rtts[int(rng.integers(0, len(rtts)))]
    q = float(rng.uniform(*s["queue_bdp_factor"]["uniform"]))
    lm = s["loss_rate_mixture"]
    if rng.random() < lm["p_none"]:
        loss = 0.0
    else:
        loss = float(rng.uniform(*lm["uniform"]))
    flows_opts = s["n_flows"]["choices"]
    n_flows = int(flows_opts[int(rng.integers(0, len(flows_opts)))])
    variation = {"type": "none"}
    if rng.random() < s["sine"]["p"]:
        amp = float(rng.uniform(*s["sine"]["amplitude_frac"]))
        per = float(rng.uniform(*s["sine"]["period_s"]))
        variation = {"type": "sine", "amplitude_frac": amp, "period_s": per}
    # base_rtt must be a multiple of 2*dt (10 ms); round to nearest 10
    rtt = int(round(rtt / 10.0) * 10)
    return {
        "name": f"train_ep{episode}",
        "duration_s": float(cfg["episode_s"]),
        "dt_ms": float(cfg.get("dt_ms", 5)),
        "bandwidth_mbps": round(bw, 3),
        "base_rtt_ms": float(max(rtt, 20)),
        "queue_bdp_factor": round(q, 3),
        "loss_rate": round(loss, 6),
        "n_flows": n_flows,
        "warmup_s": float(cfg.get("warmup_s", 2.0)),
        "bw_variation": variation,
        "seed": episode,   # link noise varies per episode
    }
