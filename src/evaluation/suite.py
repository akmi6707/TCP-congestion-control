"""Parallel experiment-suite execution.

Workers execute (scenario, controller, seed) runs and return record dicts;
the parent persists every record (including failures) sequentially, so the
index is append-only and race-free. All runs within a suite share one
experiment id; records also carry the suite name for filtering.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import io as uio
from .runner import run_one

log = logging.getLogger("suite")


def _worker_init() -> None:
    # avoid thread oversubscription in parallel workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _worker(args) -> Dict:
    (scenario, controller, seed, experiment, suite, results_dir, spec) = args
    return run_one(
        scenario=scenario, controller=controller, seed=seed,
        experiment=experiment, suite=suite,
        results_dir=Path(results_dir),
        policy_path=spec.get("policy_path"),
        ml_path=spec.get("ml_path"),
        epsilon=spec.get("epsilon", 0.0),
        reward_weights=spec.get("reward_weights"),
        save_series=bool(spec.get("save_series", False)),
        code_version=spec.get("code_version", ""),
        persist=False,
    )


def run_suite(scenarios: List[Dict], controllers: Dict[str, Dict],
              seeds: List[int], experiment: str, suite: str = "",
              results_dir: Path = uio.RESULTS_DIR,
              n_workers: int = 8, save_series_controllers: Optional[List[str]] = None,
              code_version: str = "") -> List[Dict]:
    """Run scenarios x controllers x seeds in parallel; persist all records.

    ``controllers`` maps controller name -> spec dict (policy_path, ml_path,
    epsilon, reward_weights). save_series_controllers: names for which to
    persist per-run time series (used for figures).
    """
    save_series_controllers = save_series_controllers or []
    jobs = []
    for sc in scenarios:
        for cname, spec in controllers.items():
            for seed in seeds:
                spec2 = dict(spec)
                spec2["save_series"] = cname in save_series_controllers
                jobs.append((sc, cname, int(seed), experiment, suite,
                             str(results_dir), spec2))
    log.info("suite '%s': %d runs (%d scenarios x %d controllers x %d seeds)",
             suite or experiment, len(jobs), len(scenarios), len(controllers), len(seeds))
    if n_workers <= 1:
        _worker_init()
        records = [_worker(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            records = pool.map(_worker, jobs, chunksize=1)
    # persist sequentially in the parent (append-only index, race-free)
    for rec in records:
        try:
            uio.write_run_record(rec, Path(results_dir))
        except Exception as exc:  # noqa: BLE001
            log.error("failed to persist run %s: %s", rec.get("run_id"), exc)
            rec["persist_error"] = str(exc)
    ok = sum(1 for r in records if r.get("status") == "ok")
    failed = len(records) - ok
    log.info("suite '%s' done: %d ok, %d failed", suite or experiment, ok, failed)
    return records


def suite_experiment_id(suite: str) -> str:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{suite}_{ts}"
