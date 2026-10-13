# Hybrid ML + RL TCP Congestion Control: Implementation Report

> STATUS: results sections are generated from executed experiments. All
> numbers below marked with [PENDING] are placeholders to be replaced by
> the analysis outputs; nothing in the Results section is hand-written.

## 1. Environment and architecture

### Initial directory state
The working directory `/home/akmi/Desktop/TCP-congestion-control` was
**empty** (verified: no hidden files, no repository, no datasets, no
models). Nothing had to be preserved. Full audit in `ENVIRONMENT.md`.

### OS and tools
Pop!_OS 22.04 (Ubuntu-based), Linux 7.1.1, 16 CPUs, 14 GiB RAM. Python
3.10.12, pip 22.0.2, git 2.34.1. Docker, Mininet, ns-3: absent. `tc`
present but unusable: **no root** ("no-new-privileges"), so kernel-level
netem/namespace experiments were infeasible. PyPI access worked; a venv was
bootstrapped without ensurepip via `pip3 --target` (all dependencies live
inside the project; see `scripts/setup_env.sh`).

### Selected stack and reasons
- **Custom discrete-time TCP dumbbell simulator** (pure Python): the only
  scientifically valid option without root/ns-3/Mininet; gives full
  control of the RL training loop. dt=5 ms ticks, packet-batch semantics,
  per-flow congestion windows, shared drop-tail queue, per-packet random
  loss, time-varying capacity/RTT/loss events, SACK-style recovery,
  RFC 6298 RTO, ACK-clocking via drain-spread ACKs.
- **scikit-learn** HistGradientBoosting (CPU-friendly, mature) for the ML
  component; **PyTorch CPU** for a small double-DQN; **scipy** for
  statistics; **matplotlib** for figures; **pytest** for tests.

### ML model
Three HistGradientBoostingRegressor models (max_iter=300, lr=0.08) predict
the current control interval's mean queueing delay (ms), retransmission
event rate ("loss signal"), and delivered rate (Mbps) from the 18
pre-interval sender-observable features. Held-out (scenario-disjoint)
validation: [PENDING - filled from results/ml_data/ml_metrics.json].
Model size [PENDING] bytes; single-sample inference ~3-4 ms (measured).

### RL algorithm
Double DQN, MLP 2x128 ReLU, Adam lr 7e-4, gamma 0.99, batch 128, replay
300k, n-step returns (n=3, added because 1-step TD misattributes credit
across the ~1-RTT feedback delay - observed as validation-curve collapse),
target sync 500, grad clip 10. Checkpoint = best 3-validation moving
average. Training: 900 episodes, 10 s each, scenarios sampled from
`configs/curriculum.yaml` (bw 5-60 Mbps log-uniform, RTT 20-140 ms,
queue 0.3-3 BDP, loss 0-0.5%, 1-4 flows, 30% time-varying links).

### Observation space (18 features + 3 ML predictions)
Sender-observable only; definitions and clips in `src/environment/features.py`:
rates (log2-scaled), bandwidth estimate, min-RTT, queueing delay, RTT
ratio, cwnd, BDP estimate, window overshoot, inflight fraction, loss/dupack
flags, utilization, trends, previous action and reward.

### Action space (8 discrete, bounded)
cwnd x {0.5, 0.8, 1.0, 1.02, 1.1, 1.3}; cwnd = {1.0, 1.5} x BDP estimate.
Post-action clip to [1, 4096] packets; no invalid actions possible.
25 ms control interval; bootstrap ramp until the first RTT sample +
delivered data.

### Reward (pre-registered)
r = 1.0*log2(1+g) - 0.5*min(qd,200ms)/20ms - 1.0*min(L,0.05)/0.01 -
0.05*|log2(mult)|, with g = interval goodput (Mbps), qd = mean queueing
delay, L = retransmission-event rate (all sender-observable).
Ablations zero the delay or loss term.

### Hybrid integration
The ML predictions are appended to the RL state (policy inputs). The
hybrid policy is warm-started from the final rl_only policy and
fine-tuned (the cold-start hybrid collapsed to a frozen constant policy -
documented in `configs/evaluation_protocol.yaml`; both artifacts kept).

## 2. Files created
See README.md repository map. No unexpected pre-existing files were found.

## 3. Experiments executed
[PENDING - filled from results/index.csv counts and suite experiment ids]

## 4. Quantitative results
[PENDING - filled from results/analysis/summary_*.md]

## 5. Ablation and generalization
[PENDING]

## 6. Tests
57 tests, all passing (`pytest tests/ -q`). Failures encountered during
development and their fixes are documented in code comments and git
history; final state: 57 passed, 0 failed.

## 7. Reproduction guide
See README.md ("Reproduction: exact commands").

## 8. Limitations
See docs/LIMITATIONS.md (simulation abstraction, baseline fidelity,
statistical caveats, learning-controller caveats, what is not claimed).

## 9. Research conclusion
[PENDING]
