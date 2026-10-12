"""Scenario and experiment configuration loading + validation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _check(isinstance(cfg, dict), f"top-level YAML must be a mapping: {path}")
    return cfg


MSS_BYTES = 1500          # full packet size on the wire (incl. headers)
PAYLOAD_BYTES = 1460       # TCP payload per packet (MSS - 40B IPv6/TCP hdrs)


def validate_scenario(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a scenario config; returns a normalized deep copy.

    Required keys: name, duration_s, bandwidth_mbps, base_rtt_ms.
    Optional keys: dt_ms(5), queue_packets, queue_bdp_factor(2.0), loss_rate(0),
    n_flows(1), flow_start_offsets_s, bw_variation, capacity_events,
    rtt_events, loss_events, short_flows, warmup_s(2.0), seed(0).
    """
    c = copy.deepcopy(cfg)
    for k in ("name", "duration_s", "bandwidth_mbps", "base_rtt_ms"):
        _check(k in c, f"scenario '{cfg.get('name', '?')}': missing key '{k}'")
    c.setdefault("dt_ms", 5.0)
    c.setdefault("queue_bdp_factor", 2.0)
    c.setdefault("loss_rate", 0.0)
    c.setdefault("n_flows", 1)
    c.setdefault("warmup_s", 2.0)
    c.setdefault("seed", 0)
    c["queue_packets"] = c.get("queue_packets")
    c["flow_start_offsets_s"] = c.get("flow_start_offsets_s")
    c["bw_variation"] = c.get("bw_variation") or {"type": "none"}
    c["capacity_events"] = c.get("capacity_events") or []
    c["rtt_events"] = c.get("rtt_events") or []
    c["loss_events"] = c.get("loss_events") or []
    c["short_flows"] = c.get("short_flows")

    _check(isinstance(c["name"], str) and c["name"], "name must be a non-empty string")
    _check(c["duration_s"] > 0, "duration_s must be > 0")
    _check(c["bandwidth_mbps"] > 0, "bandwidth_mbps must be > 0")
    _check(c["base_rtt_ms"] > 0, "base_rtt_ms must be > 0")
    _check(c["dt_ms"] > 0, "dt_ms must be > 0")
    _check(0.0 <= c["loss_rate"] <= 0.2, "loss_rate must be in [0, 0.2]")
    _check(c["n_flows"] >= 1, "n_flows must be >= 1")
    _check(0.0 <= c["warmup_s"] < c["duration_s"], "warmup_s must be in [0, duration_s)")

    # base_rtt must be a multiple of 2*dt (symmetric forward/backward delays
    # that land exactly on tick boundaries).
    _check(
        abs((c["base_rtt_ms"] / (2 * c["dt_ms"])) - round(c["base_rtt_ms"] / (2 * c["dt_ms"]))) < 1e-9,
        f"base_rtt_ms ({c['base_rtt_ms']}) must be a multiple of 2*dt_ms ({2 * c['dt_ms']})",
    )
    for ev in c["rtt_events"]:
        _check("t_s" in ev and "base_rtt_ms" in ev, "rtt_events need t_s and base_rtt_ms")
        _check(
            abs((ev["base_rtt_ms"] / (2 * c["dt_ms"])) - round(ev["base_rtt_ms"] / (2 * c["dt_ms"]))) < 1e-9,
            "rtt_event base_rtt_ms must be a multiple of 2*dt_ms",
        )

    bv = c["bw_variation"]
    _check(bv.get("type") in ("none", "sine", "step"), "bw_variation.type must be none|sine|step")
    if bv["type"] == "sine":
        _check(0.0 <= bv.get("amplitude_frac", 0.3) < 1.0, "sine amplitude_frac in [0,1)")
        _check(bv.get("period_s", 10.0) > 0, "sine period_s must be > 0")
    for ev in c["capacity_events"]:
        _check("t_s" in ev and "bandwidth_mbps" in ev, "capacity_events need t_s and bandwidth_mbps")
        _check(0 <= ev["t_s"] < c["duration_s"], "capacity_event t_s out of range")
        _check(ev["bandwidth_mbps"] > 0, "capacity_event bandwidth_mbps must be > 0")
    for ev in c["loss_events"]:
        _check("t_s" in ev and "loss_rate" in ev, "loss_events need t_s and loss_rate")
        _check(0 <= ev["loss_rate"] <= 0.5, "loss_event loss_rate out of range")

    if c["flow_start_offsets_s"] is not None:
        _check(
            len(c["flow_start_offsets_s"]) == c["n_flows"],
            "flow_start_offsets_s must have n_flows entries",
        )
        for off in c["flow_start_offsets_s"]:
            _check(0 <= off < c["duration_s"], "flow start offset out of range")

    if c["short_flows"] is not None:
        sf = c["short_flows"]
        for k in ("n", "size_pkts_range", "start_s_range"):
            _check(k in sf, f"short_flows missing key '{k}'")
        _check(sf["n"] >= 0, "short_flows.n must be >= 0")
        _check(len(sf["size_pkts_range"]) == 2 and 1 <= sf["size_pkts_range"][0] <= sf["size_pkts_range"][1],
               "short_flows.size_pkts_range must be [min,max] packet counts, min>=1")
        _check(len(sf["start_s_range"]) == 2 and 0 <= sf["start_s_range"][0] <= sf["start_s_range"][1] < c["duration_s"],
               "short_flows.start_s_range invalid")

    if c["queue_packets"] is None:
        _check(c["queue_bdp_factor"] > 0, "queue_bdp_factor must be > 0")
    else:
        _check(c["queue_packets"] > 0, "queue_packets must be > 0")
    return c


def load_scenario(path: str | Path) -> Dict[str, Any]:
    return validate_scenario(load_yaml(path))


def load_scenario_list(path: str | Path) -> List[Dict[str, Any]]:
    """Load a YAML file containing {scenarios: [...]}."""
    cfg = load_yaml(path)
    _check("scenarios" in cfg and isinstance(cfg["scenarios"], list), "file must contain 'scenarios: [...]'")
    return [validate_scenario(s) for s in cfg["scenarios"]]
