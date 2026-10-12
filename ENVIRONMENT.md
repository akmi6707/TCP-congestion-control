# Environment Audit (Phase 1)

Recorded: 2026-02-13 (session start). All facts below were verified by direct
execution, not assumed.

## Initial directory state

- Working directory: `/home/akmi/Desktop/TCP-congestion-control`
- Initial state: **empty** (`ls -la` showed only `.` and `..`; no hidden files).
  No pre-existing repository, code, dataset, virtual environment, simulator,
  testing framework, or trained model was present. Nothing had to be preserved.

## Operating system and hardware

- OS: Pop!_OS 22.04 LTS (Ubuntu-based), Linux kernel 7.1.1-76070111-generic x86_64
- CPU: 16 cores
- RAM: 14 GiB (5.7 GiB free at audit time)
- Disk: 311 GiB total on the project filesystem, ~155 GiB free

## Tools and availability

| Tool | Status |
|---|---|
| Python 3.10.12 | available (`/usr/bin/python3`) |
| pip 22.0.2 | available (`/usr/lib/python3/dist-packages/pip`) |
| Git 2.34.1 | available |
| Docker | **missing** |
| `tc` (iproute2) | binary present, but **unusable**: requires root |
| Linux netem kernel module | not found for the running kernel |
| Mininet | **missing** |
| ns-3 | **missing** |
| sudo | present but **blocked** ("no new privileges" flag is set; sudo refuses to run as root) |
| Network namespaces | require root — infeasible |
| Internet access (PyPI) | available (verified by a test download) |

## Pre-installed Python packages (system)

- numpy 2.2.6, pandas 2.3.3 (system interpreter)
- scipy, matplotlib, scikit-learn, torch, gym, pytest: **missing**

## Privileges and constraints

- **No root/sudo**: kernel-level experiments (`tc netem`, network namespaces,
  real-socket congestion-control experiments on shaped links) are **not
  feasible** in this environment. This is a hard, verified constraint.
- Filesystem: writes are permitted under the project workspace and `/tmp`.
  The user pip cache (`~/.cache/pip`) is not writable; the venv bootstrap
  (`python3 -m venv`) fails because `ensurepip` is unavailable
  (would need `apt install python3.10-venv` as root).
- Workaround (used): `python3 -m venv --without-pip .venv` plus installing
  pinned wheels with system `pip3 --target .venv/lib/python3.10/site-packages`.
  All dependencies therefore live inside the project directory.

## Chosen technology stack and rationale

1. **Simulation core**: a custom, seed-deterministic, discrete-time
   packet-batch TCP simulator in pure Python (numpy for aggregates only).
   Justification: no ns-3/Mininet/Docker is available and kernel experiments
   require root. A dumbbell-topology simulator with per-flow congestion
   windows, a shared drop-tail bottleneck queue, propagation-delay-shifted
   arrivals, ACK-clocking, fast retransmit and RTO gives the delayed-feedback
   dynamics that congestion-control research requires. This follows the
   interval-based abstraction used widely in learning-based congestion-control
   research. It is a *simulation of congestion-control dynamics*, not a kernel
   TCP implementation; all results are labeled as simulation results.
2. **ML component**: scikit-learn (`HistGradientBoostingRegressor`) — mature,
   fast, CPU-friendly, no GPU required.
3. **RL component**: PyTorch (CPU build) for a small DQN. CPU-only wheel keeps
   the dependency footprint reasonable (~200 MB); 16 cores are sufficient.
4. **Statistics**: scipy (Wilcoxon signed-rank, paired t-test, bootstrap,
   Holm correction), pandas for tables.
5. **Figures**: matplotlib.
6. **Tests**: pytest.
7. **Configs**: YAML (PyYAML).

Dependency versions are pinned in `requirements.txt`. The full environment can
be recreated with `scripts/setup_env.sh` (no root required).

## Infeasible experiments (documented, not attempted)

- Real-kernel TCP comparison on shaped links (needs `tc`/netem + root).
- Mininet/ns-3/Docker-based emulation (tools absent; no way to install ns-3
  within compute constraints, and no root for Mininet's namespace setup).
- Any experiment sending traffic to external systems (prohibited by task
  policy and irrelevant to the study).
