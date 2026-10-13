"""Run one (scenario, controller, seed) episode and persist a record.

Never overwrites: every run gets a unique run_id and its own JSON file.
Failures are recorded with status='failed' and the exception text.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Dict, Optional

from ..controllers import make_controller_factory
from ..environment.sim import simulate
from ..metrics import compute_metrics
from ..utils import io as uio


def build_factory(controller: str, seed: int = 0,
                  policy_path: Optional[str] = None,
                  ml_path: Optional[str] = None,
                  epsilon: float = 0.0,
                  reward_weights: Optional[Dict] = None):
    if controller in ("rl_only", "rl_only_nodelay", "rl_only_noloss",
                      "hybrid", "ml_only", "random"):
        base = "rl_only" if controller.startswith("rl_only") else controller
        return make_controller_factory(
            base, policy_path=policy_path, ml_path=ml_path,
            epsilon=epsilon, seed=seed, reward_weights=reward_weights)
    return make_controller_factory(controller)


def run_one(scenario: Dict, controller: str, seed: int,
            experiment: str, suite: str = "",
            results_dir: Path = uio.RESULTS_DIR,
            policy_path: Optional[str] = None,
            ml_path: Optional[str] = None,
            epsilon: float = 0.0,
            reward_weights: Optional[Dict] = None,
            save_series: bool = False,
            code_version: str = "",
            persist: bool = True) -> Dict:
    """Execute one run and return the record dict (also persisted)."""
    run_id = uio.new_run_id("run")
    record = {
        "run_id": run_id,
        "experiment": experiment,
        "suite": suite,
        "controller": controller,
        "seed": int(seed),
        "scenario": scenario,
        "duration_s": scenario.get("duration_s"),
        "status": "ok",
        "code_version": code_version,
        "model_info": {},
        "training_time_s": None,
    }
    t0 = time.perf_counter()
    try:
        factory = build_factory(controller, seed=seed, policy_path=policy_path,
                                ml_path=ml_path, epsilon=epsilon,
                                reward_weights=reward_weights)
        res = simulate(scenario, factory, seed=seed)
        metrics = compute_metrics(res)
        record["metrics"] = metrics
        record["wall_time_s"] = time.perf_counter() - t0
        record["inference"] = factory.inference_stats()
        record["model_info"] = factory.controller_info()
        if policy_path:
            record["model_info"]["policy_path"] = str(policy_path)
        if ml_path:
            record["model_info"]["ml_path"] = str(ml_path)
        if save_series:
            series = {
                "t_s": res.buckets_t,
                "goodput_mbps": (res.bucket_agg["goodput_bytes"] * 8 / 0.1 / 1e6).tolist(),
                "throughput_mbps": (res.bucket_agg["throughput_bytes"] * 8 / 0.1 / 1e6).tolist(),
                "queue_pkts_mean": res.bucket_agg["queue_pkts_mean"].tolist(),
                "cap_mbps_mean": res.bucket_agg["cap_mbps_mean"].tolist(),
                "loss_pkts": res.bucket_agg["loss_pkts"].tolist(),
                "per_flow": [
                    {"cwnd_mean": pf["cwnd_mean"].tolist(),
                     "goodput_bytes": pf["goodput_bytes"].tolist(),
                     "qd_mean": pf["qd_mean"].tolist(),
                     "srtt_mean": pf["srtt_mean"].tolist()}
                    for pf in res.bucket_per_flow
                ],
            }
            record["series_path"] = str(uio.write_time_series(
                run_id, series, results_dir))
    except Exception as exc:  # noqa: BLE001 - record all failures
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=8)
        record["wall_time_s"] = time.perf_counter() - t0
    # standalone use persists immediately; suite workers return records and
    # the parent persists them sequentially (single-writer, race-free)
    if persist:
        try:
            uio.write_run_record(record, results_dir)
        except Exception as exc:  # noqa: BLE001
            record["persist_error"] = str(exc)
    return record
