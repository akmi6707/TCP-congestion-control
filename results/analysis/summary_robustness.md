# Results summary: robustness_20261013T154212

runs ok: 200, failed: 0

| controller | goodput (Mbps) | mean RTT (ms) | p95 RTT (ms) | loss (%) | Jain |
|---|---|---|---|---|---|
| reno | 10.08 | 128.4 | 410.0 | 2.03 | 1.000 |
| cubic | 8.36 | 146.7 | 219.2 | 3.84 | 1.000 |
| bbr_like | 16.34 | 83.4 | 141.2 | 0.03 | 1.000 |
| vegas_like | 18.08 | 116.2 | 176.0 | 0.05 | 1.000 |
| random | 6.52 | 78.9 | 115.6 | 0.03 | 1.000 |
| ml_only | 6.38 | 78.4 | 102.0 | 0.04 | 1.000 |
| rl_only | 16.38 | 90.2 | 118.4 | 0.02 | 1.000 |
| hybrid | 16.43 | 86.7 | 118.9 | 0.01 | 1.000 |
| rl_only_nodelay | 1.57 | 718.0 | 3824.6 | 20.60 | 1.000 |
| rl_only_noloss | 16.81 | 94.7 | 134.9 | 0.02 | 1.000 |

## Paired tests (primary controller vs others, goodput, one-sided Wilcoxon over (scenario, seed) pairs)

| comparison | n | mean A | mean B | diff | p (raw) | Holm reject | Cliff's delta | rank-biserial | geo-ratio [CI] |
|---|---|---|---|---|---|---|---|---|---|
| reno | 20 | 16.43 | 10.08 | +6.36 | 0.0000 | True | +1.00 | +1.00 | 1.5510593075101915 [1.2835937503601826, 1.7985209644204887] |
| cubic | 20 | 16.43 | 8.36 | +8.08 | 0.0000 | True | +1.00 | +1.00 | 1.8090310232087556 [1.2674773815783202, 2.305991618358924] |
| bbr_like | 20 | 16.43 | 16.34 | +0.10 | 0.1741 | False | +0.50 | +0.24 | 0.9591332500067448 [0.7513731453177181, 1.1404141033809916] |
| vegas_like | 20 | 16.43 | 18.08 | -1.65 | 0.5000 | False | +0.00 | +0.00 | 0.8658224592422357 [0.5515405693841186, 1.1555138162094596] |
| random | 20 | 16.43 | 6.52 | +9.91 | 0.0000 |  | +1.00 | +1.00 | 2.4562294088390875 [2.2787641428870384, 2.6475152892315283] |
| ml_only | 20 | 16.43 | 6.38 | +10.05 | 0.0000 |  | +1.00 | +1.00 | 2.5345338090792633 [1.836500283200595, 3.107559848012738] |
| rl_only | 20 | 16.43 | 16.38 | +0.05 | 0.2153 |  | -0.20 | +0.20 | 1.0137012123041598 [0.9817691838607329, 1.0466718295088493] |
| rl_only_nodelay | 20 | 16.43 | 1.57 | +14.87 | 0.0000 |  | +1.00 | +1.00 | 9.91275763464916 [8.607358733608367, 11.310741870045279] |
| rl_only_noloss | 20 | 16.43 | 16.81 | -0.38 | 0.9994 |  | -0.90 | -0.82 | 0.9733511738678393 [0.9595241082796863, 0.9873774921283635] |
| rl_only_p95rtt | 20 | 118.88 | 118.39 | +0.50 | 0.0294 |  | -0.55 | -0.55 | None [None, None] |

## Guardrails (aggregate means; ok=False = violated)

- vs reno: m_mean_rtt_ms = 86.685 vs 128.431 -> OK
- vs reno: m_p95_rtt_ms = 118.883 vs 409.995 -> OK
- vs reno: m_loss_rate = 0.000 vs 0.020 -> OK
- vs reno: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs cubic: m_mean_rtt_ms = 86.685 vs 146.683 -> OK
- vs cubic: m_p95_rtt_ms = 118.883 vs 219.162 -> OK
- vs cubic: m_loss_rate = 0.000 vs 0.038 -> OK
- vs cubic: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs bbr_like: m_mean_rtt_ms = 86.685 vs 83.450 -> OK
- vs bbr_like: m_p95_rtt_ms = 118.883 vs 141.244 -> OK
- vs bbr_like: m_loss_rate = 0.000 vs 0.000 -> OK
- vs bbr_like: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs vegas_like: m_mean_rtt_ms = 86.685 vs 116.183 -> OK
- vs vegas_like: m_p95_rtt_ms = 118.883 vs 175.998 -> OK
- vs vegas_like: m_loss_rate = 0.000 vs 0.001 -> OK
- vs vegas_like: m_jain_fairness = 1.000 vs 1.000 -> OK
