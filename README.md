# Hybrid ML + RL TCP Congestion Control (Research Prototype)

An experimentally validated research prototype studying **"Intelligent TCP
Congestion Control Using Hybrid Machine Learning and Reinforcement
Learning"** in a controlled simulator study. All results in this repository
come from actually executed experiments; nothing is simulated by hand or
fabricated. See `docs/LIMITATIONS.md` for what the simulation does and does
not model, and `REPORT.md` for the full findings.

## What this is

- A **discrete-time TCP dumbbell simulator** (Python, 5 ms ticks, dt-based,
  deterministic per (scenario, seed)) with per-flow congestion windows,
  ACK-clocking, a shared drop-tail bottleneck queue, random loss,
  time-varying bandwidth/RTT, SACK-style loss recovery, and RFC 6298 RTO.
- **Conventional baselines**: Reno (RFC 5681-style), CUBIC (RFC 8312-style),
  BBR-inspired, Vegas-family delay-based heuristic.
- **Learning controllers**:
  - `random` - uniform-action floor baseline
  - `rl_only` - double-DQN over 18 sender-observable features
  - `hybrid` - same RL policy, state augmented with 3 ML predictions
    (next-interval queueing delay, loss signal, rate from 3
    HistGradientBoosting models)
  - `ml_only` - supervised-only controller
  - reward ablations `rl_only_nodelay` / `rl_only_noloss`
- **Pre-registered evaluation protocol** (`configs/evaluation_protocol.yaml`):
  goodput is primary; mean/p95 RTT, loss, Jain fairness are guardrails;
  paired (scenario, seed) one-sided Wilcoxon with Holm correction; effect
  sizes (Cliff's delta, rank-biserial) and scenario-cluster bootstrap CIs.
- **Experiment suites** (`configs/scenarios_*.yaml`): 13-scenario core
  matrix (bandwidth x RTT x queue x loss x flows x time-varying), 5
  held-out generalization scenarios (outside the training ranges), 4
  abrupt-change robustness scenarios, and a short-flow workload.

## Setup (no root required)

```bash
bash scripts/setup_env.sh      # creates .venv, installs pinned deps incl. CPU torch
.venv/bin/python3 -c "import torch, sklearn, scipy; print('ok')"
```

(See `ENVIRONMENT.md` for the full environment audit; the venv is built
without ensurepip, which is unavailable without root here.)

## Reproduction: exact commands

```bash
export MPLCONFIGDIR=/tmp/mplcache

# 1. Tests (57 tests: units + integration)
.venv/bin/python3 -m pytest tests/ -q

# 2. Simulator/baseline smoke test (Phase-2 gate)
.venv/bin/python3 scripts/smoke_test.py

# 3. ML pipeline: collect training data (instrumented baselines + random
#    controller over 24 curriculum scenarios) and train the predictor
.venv/bin/python3 scripts/collect_ml_data.py --n-scenarios 24 --episode-s 12 --workers 12
.venv/bin/python3 scripts/train_ml.py

# 4. RL training (each ~6-8 min except hybrid ~40 min; run 4 in parallel)
.venv/bin/python3 scripts/train_rl.py --variant rl_only   --seed 0
.venv/bin/python3 scripts/train_rl.py --variant hybrid     --seed 0
.venv/bin/python3 scripts/train_rl.py --variant rl_only_nodelay --seed 0
.venv/bin/python3 scripts/train_rl.py --variant rl_only_noloss  --seed 0

# 5. Experiment suites (8 controllers x scenarios x 5 seeds, parallel)
.venv/bin/python3 scripts/run_experiments.py --suite core          --workers 12
.venv/bin/python3 scripts/run_experiments.py --suite generalization --workers 12
.venv/bin/python3 scripts/run_experiments.py --suite robustness     --workers 12
.venv/bin/python3 scripts/run_experiments.py --suite workload      --workers 12

# 6. Statistical analysis (writes results/analysis/*.json + summary markdown)
.venv/bin/python3 scripts/analyze_results.py --suite core
.venv/bin/python3 scripts/analyze_results.py --suite generalization
.venv/bin/python3 scripts/analyze_results.py --suite robustness

# 7. Figures (results/ -> figures/*.png)
.venv/bin/python3 scripts/make_figures.py
```

Models land in `models/` (`ml_predictor.joblib`, `dqn_<variant>.pt`), run
records in `results/runs/*.json` (one JSON per run; append-only index at
`results/index.csv`), ML dataset in `results/ml_data/`, training logs in
`results/training/`, analysis in `results/analysis/`, figures in
`figures/`.

To re-run a single scenario manually:

```bash
.venv/bin/python3 - <<'EOF'
import sys; sys.path.insert(0, ".")
from src.evaluation.runner import run_one
sc = {"name": "demo", "duration_s": 20, "bandwidth_mbps": 20,
      "base_rtt_ms": 60, "queue_bdp_factor": 2.0, "loss_rate": 0.0,
      "n_flows": 1, "warmup_s": 2.0, "seed": 0}
rec = run_one(sc, "hybrid", seed=0, experiment="manual",
               policy_path="models/dqn_hybrid.pt",
               ml_path="models/ml_predictor.joblib",
               results_dir="results")
print(rec["metrics"])
EOF
```

## Repository layout

```
configs/         scenario suites, curriculum, learning hyperparameters,
                 pre-registered evaluation protocol
src/environment  simulator (sim.py) and feature pipeline (features.py)
src/controllers  Reno, CUBIC, BBR-like, Vegas-like + learning controllers
src/models       ML predictor, reward, DQN agent
src/training     curriculum sampler, instrumented data collection wrapper
src/evaluation  run/suite runners, statistical analysis
src/metrics      metric definitions and computation
src/utils        config validation, RNG, persistence, logging
scripts/         all executable pipeline steps (see above)
tests/           pytest suite (57 tests)
docs/            architecture, limitations
results/         machine-readable run records, analysis, training logs
figures/         generated figures
```

## Interpreting results

- `results/analysis/summary_core.md` - aggregate table + confirmatory
  tests (hybrid vs conventional baselines) + guardrails.
- `results/analysis/summary_generalization.md` - held-out conditions
  (descriptive).
- `figures/goodput_core.png`, `figures/pareto_core.png`,
  `figures/window_dynamics.png`, `figures/training_curves.png`,
  `figures/robustness_timeseries.png`.
- `REPORT.md` - the complete technical report with honest findings,
  including where the proposed approach does NOT win.

## Honest-results policy

Every number in `REPORT.md` is computed from persisted run records under
`results/runs/` by `scripts/analyze_results.py`. Failed runs are recorded
with status `failed` and included in counts. The evaluation protocol and
success criteria were fixed in `configs/evaluation_protocol.yaml` before
the final policies were evaluated (git history documents the ordering).
