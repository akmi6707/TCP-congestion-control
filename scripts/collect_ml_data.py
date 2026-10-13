#!/usr/bin/env python3
"""Collect ML training data: (features_t -> outcomes_{t+1}) rows from
conventional baselines (instrumented) and a uniform-random controller,
across scenarios sampled from the training distribution.

Rows are written to results/ml_data/ml_dataset.npz with a scenario-disjoint
train/val split (split by scenario group, never by row).
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from src.environment.sim import simulate
from src.controllers import make_controller_factory
from src.controllers.learning import make_learning_controller_factory
from src.training.instrument import InstrumentedController
from src.training.curriculum import sample_scenario
from src.utils.config import load_yaml

log = logging.getLogger("collect_ml")

_G = {"interval_ticks": 5, "dt_ms": 5.0}


def _init_worker(interval_ticks, dt_ms):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _G["interval_ticks"] = interval_ticks
    _G["dt_ms"] = dt_ms


def worker(args):
    sc, controller_name, seed = args
    interval_ticks, dt_ms = _G["interval_ticks"], _G["dt_ms"]
    rows_X, rows_Y, meta = [], [], []

    def collect(features, stats, flow, controller):
        rows_X.append(features)
        rows_Y.append([stats.qd_mean_ms if stats.qd_mean_ms is not None else 0.0,
                       stats.loss_signal,
                       stats.rate_mbps])

    if controller_name in ("reno", "cubic", "bbr_like", "vegas_like"):
        def factory(flow_id, _name=controller_name):
            inner = make_controller_factory(_name)(flow_id)
            return InstrumentedController(flow_id, inner, interval_ticks, dt_ms,
                                          collect_fn=collect)
    else:  # random learning controller: real feature pipeline + actions
        f = make_learning_controller_factory("random", seed=seed,
                                             collect_fn=collect)
        factory = f.build_fn

    simulate(sc, factory, seed=seed)
    for _ in range(len(rows_X)):
        meta.append((sc["name"], controller_name))
    return (np.asarray(rows_X, dtype=np.float32),
            np.asarray(rows_Y, dtype=np.float32),
            meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/curriculum.yaml")
    ap.add_argument("--n-scenarios", type=int, default=24)
    ap.add_argument("--episode-s", type=float, default=12.0)
    ap.add_argument("--seed-base", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/ml_data/ml_dataset.npz")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_yaml(args.config)
    dt_ms = float(cfg.get("dt_ms", 5))
    interval_ticks = 5

    controllers = ["reno", "cubic", "bbr_like", "vegas_like", "random"]
    jobs = []
    for i in range(args.n_scenarios):
        sc = sample_scenario(cfg, i, args.seed_base)
        sc["duration_s"] = args.episode_s
        sc["name"] = f"mldata_{i}"
        for c in controllers:
            jobs.append((sc, c, i))
    log.info("collecting %d runs (%d scenarios x %d controllers)",
             len(jobs), args.n_scenarios, len(controllers))

    if args.workers <= 1:
        _init_worker(interval_ticks, dt_ms)
        results = [worker(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(interval_ticks, dt_ms)) as pool:
            results = pool.map(worker, jobs, chunksize=1)

    Xs, Ys, meta = [], [], []
    for X, Y, m in results:
        Xs.append(X)
        Ys.append(Y)
        meta.extend(m)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    groups = np.array([int(m[0].split("_")[-1]) for m in meta], dtype=np.int64)
    ctrl_names = [m[1] for m in meta]

    # scenario-disjoint split: every 3rd scenario group is validation
    val_groups = set(range(0, args.n_scenarios, 3))
    split = np.array([1 if g in val_groups else 0 for g in groups], dtype=np.int8)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, X=X, Y=Y, groups=groups, split=split,
        ctrl_names=np.array(ctrl_names),
        feature_desc="see src/environment/features.py",
    )
    log.info("saved %s: %d rows (%d train / %d val), X shape %s",
             out, X.shape[0], int((split == 0).sum()), int((split == 1).sum()), X.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
