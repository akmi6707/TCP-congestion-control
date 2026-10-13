# Limitations and Threats to Validity

## Simulation abstraction (most important)

All results are SIMULATION results from a custom discrete-time (5 ms tick)
packet-batch dumbbell simulator, not kernel TCP:

- No packet-level pacing; senders are window-limited with a per-tick burst
  cap of 2x link service (models NIC/TSO granularity). ACKs are drained
  through the bottleneck and spread across its service window.
- Loss within a batch is modeled as one contiguous hole (position from a
  seeded uniform); overflow drops are contiguous tail runs of the arrival
  stream with per-tick rotation to avoid flow bias.
- Loss recovery is shared by all controllers: fast retransmit on 3
  dupacks, SACK-guided run retransmission (bounded, rate-limited), and a
  bounded no-SACK fallback. This resembles modern RACK/SACK TCP more than
  1990s NewReno; results for loss-based baselines depend on this choice.
- Delayed ACKs, ACK compression, header overhead, and reverse-path
  congestion are not modeled. Goodput counts 1460 payload bytes per
  1500-byte packet.
- Measured RTT carries a ~1-3 ms self-drain floor from per-tick ACK
  clusters at 5 ms granularity; delay-based thresholds below ~3 ms are
  therefore not meaningful.

## Baseline fidelity

- **reno**: RFC 5681-style slow start + AIMD. Slow-start overshoot on deep
  buffers causes large loss bursts - realistic, but exact magnitudes depend
  on the model above.
- **cubic**: RFC 8312-style with C=0.4, beta=0.7; no HyStart, no
  ack-decoupling. Its slow recovery after big losses at high BDP is
  RFC-consistent but slightly pessimistic vs Linux (which has HyStart).
- **bbr_like**: BBRv1-inspired (documented divergences in
  `src/controllers/bbr_like.py`): window-only emulation of pacing,
  running min-RTT, round-based max-bandwidth filter with a 40-round
  window, 2xRTT probe phases. NOT kernel BBR; do not cite as BBR.
- **vegas_like**: delay-threshold (5/15 ms) variant of the Vegas family
  (the packet-diff form starves in the presence of the RTT floor above).

## Statistical caveats

- Pairing: identical (scenario, seed) gives identical link-noise draw
  sequences across controllers, but which packets they apply to still
  depends on controller behavior; pairing is approximate.
- 5 seeds per scenario; the per-scenario paired tests (n=5) are weak; the
  confirmatory test pools (scenario, seed) pairs across the core suite
  (n=65) which pools within-scenario correlation into one test - a
  known simplification, mitigated by also reporting per-scenario means
  with CIs and geometric-mean ratios with scenario-cluster bootstrap CIs.
- The Holm correction is applied across the 4 primary conventional
  comparisons, as pre-registered.

## Learning-controller caveats

- Single training seed (seed 0) for the reported DQN policies;
  run-to-run training variance is not quantified (documented gap).
- The ML predictor's held-out R^2 for queueing delay is only ~0.36 (bursty
  target); the hybrid's benefit must be interpreted with that in mind.
- Inference cost: the HGB predictor costs ~4 ms per control step
  (single-sample sklearn predict x3) vs ~0.06 ms for the DQN alone -
  17% of the 25 ms control budget. Real deployments would batch or
  compile this.
- Training used the same simulator family as evaluation (in-distribution
  core suite by design); the generalization suite probes extrapolation,
  but simulation-to-real transfer is entirely untested (kernel
  integration was infeasible: no root, no ns-3/Mininet - see
  ENVIRONMENT.md).

## What is NOT claimed

- No claim of kernel-TCP equivalence, real-network validation, or
  readiness for production.
- No claim that the conventional baselines represent their Linux
  counterparts' absolute performance; they are algorithm-level
  implementations under a shared transport model.
- The scenario space (5-100 Mbps, 20-200 ms, <=1% loss) covers only a
  fraction of real Internet conditions; no WiFi/cellular loss models,
  no buffer sizes below 0.25 BDP, no ECN.
