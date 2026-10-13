# Results summary: generalization_20261013T154145

runs ok: 250, failed: 0

| controller | goodput (Mbps) | mean RTT (ms) | p95 RTT (ms) | loss (%) | Jain |
|---|---|---|---|---|---|
| reno | 4.48 | 142.5 | 176.6 | 6.26 | 1.000 |
| cubic | 4.66 | 165.4 | 219.1 | 6.45 | 1.000 |
| bbr_like | 40.12 | 136.6 | 205.1 | 0.44 | 1.000 |
| vegas_like | 40.57 | 126.9 | 148.1 | 1.07 | 1.000 |
| random | 13.30 | 133.4 | 185.0 | 0.20 | 1.000 |
| ml_only | 10.44 | 147.4 | 204.7 | 0.19 | 1.000 |
| rl_only | 38.40 | 145.3 | 200.0 | 0.50 | 1.000 |
| hybrid | 38.13 | 138.1 | 194.4 | 0.47 | 1.000 |
| rl_only_nodelay | 13.61 | 414.2 | 1976.2 | 14.04 | 1.000 |
| rl_only_noloss | 39.56 | 143.9 | 219.1 | 0.45 | 1.000 |

## Paired tests (primary controller vs others, goodput, one-sided Wilcoxon over (scenario, seed) pairs)

| comparison | n | mean A | mean B | diff | p (raw) | Holm reject | Cliff's delta | rank-biserial | geo-ratio [CI] |
|---|---|---|---|---|---|---|---|---|---|
| reno | 25 | 38.13 | 4.48 | +33.65 | 0.0000 | True | +0.92 | +0.99 | 7.7987912290242525 [1.75593199293025, 34.637528605199] |
| cubic | 25 | 38.13 | 4.66 | +33.47 | 0.0000 | True | +1.00 | +1.00 | 4.761223890578483 [2.384885454812275, 10.84189174840579] |
| bbr_like | 25 | 38.13 | 40.12 | -1.99 | 0.5803 | False | +0.20 | -0.05 | 0.8967543334042366 [0.692706154645777, 1.0823034459816143] |
| vegas_like | 25 | 38.13 | 40.57 | -2.44 | 0.6913 | False | +0.12 | -0.11 | 1.416720508999084 [0.9510121171978064, 2.1350995746359405] |
| random | 25 | 38.13 | 13.30 | +24.83 | 0.0000 |  | +0.60 | +0.85 | 1.8390131262633111 [1.0869969646008746, 2.9002483338421854] |
| ml_only | 25 | 38.13 | 10.44 | +27.69 | 0.0018 |  | +0.20 | +0.66 | 1.9337349304417206 [0.979265277219595, 3.912318045679491] |
| rl_only | 25 | 38.13 | 38.40 | -0.27 | 0.8954 |  | -0.52 | -0.29 | 1.1059014288196485 [0.9411286722249695, 1.4774012457214643] |
| rl_only_nodelay | 25 | 38.13 | 13.61 | +24.52 | 0.0001 |  | +0.76 | +0.83 | 2.0358442243732595 [1.0270198647933257, 3.1646486309623105] |
| rl_only_noloss | 25 | 38.13 | 39.56 | -1.43 | 0.9658 |  | -0.20 | -0.42 | 1.0313129973164685 [0.6661483039592916, 1.6956163349515976] |
| rl_only_p95rtt | 25 | 194.41 | 200.02 | -5.61 | 0.0251 |  | -0.60 | -0.45 | None [None, None] |

## Guardrails (aggregate means; ok=False = violated)

- vs reno: m_mean_rtt_ms = 138.090 vs 142.484 -> OK
- vs reno: m_p95_rtt_ms = 194.409 vs 176.594 -> VIOLATION
- vs reno: m_loss_rate = 0.005 vs 0.063 -> OK
- vs reno: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs cubic: m_mean_rtt_ms = 138.090 vs 165.444 -> OK
- vs cubic: m_p95_rtt_ms = 194.409 vs 219.127 -> OK
- vs cubic: m_loss_rate = 0.005 vs 0.064 -> OK
- vs cubic: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs bbr_like: m_mean_rtt_ms = 138.090 vs 136.615 -> OK
- vs bbr_like: m_p95_rtt_ms = 194.409 vs 205.140 -> OK
- vs bbr_like: m_loss_rate = 0.005 vs 0.004 -> OK
- vs bbr_like: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs vegas_like: m_mean_rtt_ms = 138.090 vs 126.915 -> OK
- vs vegas_like: m_p95_rtt_ms = 194.409 vs 148.111 -> VIOLATION
- vs vegas_like: m_loss_rate = 0.005 vs 0.011 -> OK
- vs vegas_like: m_jain_fairness = 1.000 vs 1.000 -> OK
