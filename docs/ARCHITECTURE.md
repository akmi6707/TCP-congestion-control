# Architecture and Data/Control Flow

## System overview

```
                +--------------------------------------------------+
                |              Simulator (src/environment)         |
                |  discrete-time (dt=5ms) dumbbell bottleneck       |
                |  - per-flow cwnd, ACK-clocking, SACK recovery     |
                |  - drop-tail queue + per-packet random loss        |
                |  - time-varying capacity / RTT / loss events       |
                +-------------------------+------------------------+
                                          | per-tick obs (sender-visible)
                                          v
   +-------------------------------- Controller (src/controllers) ------+
   |  IntervalAggregator (25 ms control intervals, src/environment/    |
   |  features.py) -> IntervalStats -> build_features() (18 dims)      |
   |     [hybrid] -> MLPredictor.predict() -> 3 predictions appended   |
   |     -> policy.act(state) -> action index (8 bounded actions)       |
   |     -> _ActionSpace.apply() clips cwnd into [1, 4096] pkts       |
   +------------------------------+-----------------------------------+
                                  | cwnd
                                  v
                          simulator applies window
                                          |
              reward (per interval, sender-observable) --> DQNAgent.remember()
```

## Control pipeline per 25 ms control interval

1. **Network-state collection**: the simulator hands every controller a
   per-tick observation (new acks, dupacks, loss/RTO flags, srtt, min-rtt,
   delivered-rate estimate, cwnd, inflight). All sender-visible only.
2. **Feature preprocessing**: `IntervalAggregator` folds 5 ticks into
   `IntervalStats`; `build_features` produces the 18-dim vector (log-scaled,
   clipped; definitions in `src/environment/features.py`). Missing values
   (no RTT sample yet) trigger a safe bootstrap mode (additive ramp with
   loss backoff) instead of NaN propagation.
3. **ML inference** (hybrid only): `MLPredictor` (3 HistGradientBoosting
   models) predicts the about-to-run interval's queueing delay, loss
   signal, and rate from the 18 features; predictions are clipped and
   appended (3 dims).
4. **RL state construction**: 18-dim (rl_only) or 21-dim (hybrid) vector.
5. **Policy inference**: double-DQN MLP (2x128) argmax (epsilon=0 at
   evaluation).
6. **Action validation**: the action space contains only bounded
   multipliers (x0.5..x1.3) and BDP-anchored set-points; `_ActionSpace`
   additionally clips cwnd to [1, 4096] packets. Invalid actions cannot
   occur by construction.
7. **Action application**: cwnd change effective for the next interval.
8. **Reward and metric collection**: interval stats -> reward
   (log-throughput - queueing-delay - loss-signal - oscillation; weights in
   `configs/learning.yaml`); run-level metrics from `src/metrics`.
9. **Model updates** (training only): transitions go to a per-flow n-step
   (n=3) bookkeeping in the shared `DQNAgent`; replay buffer 300k;
   one gradient step per 2 env steps.

## ML component role

The supervised model predicts the current interval's (queueing delay, loss
signal, rate) from pre-interval features, i.e. a one-step-ahead network
forecast. In the hybrid controller these forecasts are policy inputs (the
RL policy learns to use them); in the ML-only controller they directly set
the window target (predicted rate x min-RTT x 1.15 headroom, backing off
when predicted loss > 1%). Training data: instrumented runs of Reno, CUBIC,
BBR-like, Vegas-like and a random-action controller across 24 curriculum
scenarios (scenario-disjoint train/val split).

## RL component

Double DQN (PyTorch, CPU), gamma=0.99, n-step=3 returns (the ~1-RTT
feedback delay misattributes credit under 1-step TD - observed as
validation-curve collapse during development), Huber loss, grad clip 10,
Adam lr 7e-4, target sync 500 steps, replay 300k, batch 128. Centralized
training (all flows share policy + buffer), decentralized execution.
Checkpoint selection: best 3-validation moving average of mean validation
goodput (prevents spike-chasing).

## Determinism and pairing

The link-noise draws (random-loss + hole-position uniforms) are generated
from (scenario seed, "link_noise") via a seeded numpy Generator. All
controllers in a scenario+seed see the same draw sequence at the same
times, making (scenario, seed) pairs comparable across controllers
(best-effort pairing; exact packet-level pairing is impossible because
sending patterns differ - see docs/LIMITATIONS.md).
