#!/usr/bin/env python3
"""Run an experiment suite: scenarios x controllers x seeds, in parallel.

Examples:
  .venv/bin/python3 scripts/run_experiments.py --suite core --workers 12
  .venv/bin/python3 scripts/run_experiments.py --suite generalization \
      --controllers hybrid,rl_only,reno,bbr_like --seeds 0,1,2,3,4
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.evaluation.suite import run_suite, suite_experiment_id
from src.utils import io as uio
from src.utils.config import load_scenario_list

SUITES = {
    "core": "configs/scenarios_core.yaml",
    "generalization": "configs/scenarios_generalization.yaml",
    "robustness": "configs/scenarios_robustness.yaml",
    "workload": "configs/scenarios_workload.yaml",
}

DEFAULT_CONTROLLERS = [
    "reno", "cubic", "bbr_like", "vegas_like", "random",
    "ml_only", "rl_only", "hybrid",
]


def controller_specs(names, policy_dir="models", ml_path="models/ml_predictor.joblib"):
    specs = {}
    for n in names:
        if n in ("reno", "cubic", "bbr_like", "vegas_like", "random"):
            specs[n] = {}
        elif n == "ml_only":
            specs[n] = {"ml_path": ml_path}
        elif n in ("rl_only", "rl_only_nodelay", "rl_only_noloss"):
            specs[n] = {"policy_path": f"{policy_dir}/dqn_{n}.pt"}
        elif n == "hybrid":
            specs[n] = {"policy_path": f"{policy_dir}/dqn_hybrid.pt",
                        "ml_path": ml_path}
        else:
            raise SystemExit(f"unknown controller: {n}")
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES) + ["all"])
    ap.add_argument("--controllers", default=",".join(DEFAULT_CONTROLLERS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--save-series-for", default="hybrid,reno,bbr_like",
                    help="controllers whose per-run time series are persisted")
    ap.add_argument("--experiment-id", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uio.setup_logging()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

    names = [c.strip() for c in args.controllers.split(",") if c.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    specs = controller_specs(names)

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    code_version = uio.git_commit()
    for suite in suites:
        scenarios = load_scenario_list(SUITES[suite])
        experiment = args.experiment_id or suite_experiment_id(suite)
        run_suite(scenarios, specs, seeds, experiment=experiment, suite=suite,
                  results_dir=args.results_dir, n_workers=args.workers,
                  save_series_controllers=[c for c in args.save_series_for.split(",") if c],
                  code_version=code_version)
        print(f"suite '{suite}' -> experiment id: {experiment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
