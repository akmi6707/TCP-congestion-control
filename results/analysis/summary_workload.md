# Results summary: workload_20261013T154235

runs ok: 50, failed: 0

| controller | goodput (Mbps) | mean RTT (ms) | p95 RTT (ms) | loss (%) | Jain |
|---|---|---|---|---|---|
| reno | 13.11 | 110.6 | 284.7 | 1.33 | 1.000 |
| cubic | 8.76 | 109.2 | 180.9 | 3.25 | 1.000 |
| bbr_like | 18.31 | 67.9 | 81.2 | 0.00 | 1.000 |
| vegas_like | 14.22 | 115.8 | 189.1 | 0.11 | 1.000 |
| random | 8.35 | 64.8 | 75.4 | 0.00 | 1.000 |
| ml_only | 7.43 | 62.3 | 64.7 | 0.00 | 1.000 |
| rl_only | 18.88 | 86.3 | 95.9 | 0.00 | 1.000 |
| hybrid | 18.82 | 82.2 | 93.2 | 0.00 | 1.000 |
| rl_only_nodelay | 2.11 | 984.3 | 8236.1 | 19.42 | 1.000 |
| rl_only_noloss | 18.93 | 87.4 | 98.0 | 0.00 | 1.000 |

## Paired tests (primary controller vs others, goodput, one-sided Wilcoxon over (scenario, seed) pairs)

| comparison | n | mean A | mean B | diff | p (raw) | Holm reject | Cliff's delta | rank-biserial | geo-ratio [CI] |
|---|---|---|---|---|---|---|---|---|---|
| reno | 5 | 18.82 | 13.11 | +5.72 | 0.0312 | False | +1.00 | +1.00 | 1.436200340634531 [1.436200340634531, 1.436200340634531] |
| cubic | 5 | 18.82 | 8.76 | +10.06 | 0.0312 | False | +1.00 | +1.00 | 2.148534945041922 [2.148534945041922, 2.148534945041922] |
| bbr_like | 5 | 18.82 | 18.31 | +0.52 | 0.0312 | False | +1.00 | +1.00 | 1.0281349684553769 [1.0281349684553769, 1.0281349684553769] |
| vegas_like | 5 | 18.82 | 14.22 | +4.60 | 0.0625 | False | +0.60 | +0.87 | 1.3234209902001934 [1.3234209902001934, 1.3234209902001934] |
| random | 5 | 18.82 | 8.35 | +10.47 | 0.0312 |  | +1.00 | +1.00 | 2.254679144385027 [2.254679144385027, 2.254679144385027] |
| ml_only | 5 | 18.82 | 7.43 | +11.40 | 0.0312 |  | +1.00 | +1.00 | 2.5350263921417837 [2.5350263921417837, 2.5350263921417837] |
| rl_only | 5 | 18.82 | 18.88 | -0.06 | 0.9375 |  | -0.60 | -0.73 | 0.9967836598926513 [0.9967836598926513, 0.9967836598926513] |
| rl_only_nodelay | 5 | 18.82 | 2.11 | +16.71 | 0.0312 |  | +1.00 | +1.00 | 8.916697405631378 [8.916697405631378, 8.916697405631378] |
| rl_only_noloss | 5 | 18.82 | 18.93 | -0.10 | 1.0000 |  | -1.00 | -1.00 | 0.9945008605261895 [0.9945008605261895, 0.9945008605261895] |
| rl_only_p95rtt | 5 | 93.19 | 95.93 | -2.74 | 0.0312 |  | -1.00 | -1.00 | None [None, None] |

## Guardrails (aggregate means; ok=False = violated)

- vs reno: m_mean_rtt_ms = 82.229 vs 110.553 -> OK
- vs reno: m_p95_rtt_ms = 93.192 vs 284.724 -> OK
- vs reno: m_loss_rate = 0.000 vs 0.013 -> OK
- vs reno: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs cubic: m_mean_rtt_ms = 82.229 vs 109.164 -> OK
- vs cubic: m_p95_rtt_ms = 93.192 vs 180.888 -> OK
- vs cubic: m_loss_rate = 0.000 vs 0.032 -> OK
- vs cubic: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs bbr_like: m_mean_rtt_ms = 82.229 vs 67.910 -> VIOLATION
- vs bbr_like: m_p95_rtt_ms = 93.192 vs 81.154 -> VIOLATION
- vs bbr_like: m_loss_rate = 0.000 vs 0.000 -> OK
- vs bbr_like: m_jain_fairness = 1.000 vs 1.000 -> OK
- vs vegas_like: m_mean_rtt_ms = 82.229 vs 115.808 -> OK
- vs vegas_like: m_p95_rtt_ms = 93.192 vs 189.060 -> OK
- vs vegas_like: m_loss_rate = 0.000 vs 0.001 -> OK
- vs vegas_like: m_jain_fairness = 1.000 vs 1.000 -> OK
