# Results summary: core_20261013T154027

runs ok: 650, failed: 0

| controller | goodput (Mbps) | mean RTT (ms) | p95 RTT (ms) | loss (%) | Jain |
|---|---|---|---|---|---|
| reno | 12.56 | 107.6 | 207.5 | 2.11 | 0.991 |
| cubic | 11.34 | 112.5 | 171.5 | 3.13 | 0.983 |
| bbr_like | 17.85 | 85.7 | 120.2 | 0.12 | 0.997 |
| vegas_like | 17.38 | 96.5 | 120.4 | 0.25 | 0.993 |
| random | 7.60 | 80.5 | 109.8 | 0.14 | 0.995 |
| ml_only | 7.82 | 79.6 | 91.5 | 0.04 | 0.997 |
| rl_only | 17.36 | 97.7 | 143.2 | 2.16 | 0.987 |
| hybrid | 17.86 | 80.7 | 101.9 | 0.04 | 0.988 |
| rl_only_nodelay | 4.02 | 728.2 | 5164.7 | 23.86 | 1.000 |
| rl_only_noloss | 17.97 | 96.4 | 132.3 | 0.31 | 0.975 |

## Paired tests (primary controller vs others, goodput, one-sided Wilcoxon over (scenario, seed) pairs)

| comparison | n | mean A | mean B | diff | p (raw) | Holm reject | Cliff's delta | rank-biserial | geo-ratio [CI] |
|---|---|---|---|---|---|---|---|---|---|
| reno | 65 | 17.86 | 12.56 | +5.30 | 0.0000 | True | +1.00 | +1.00 | 1.5523588828167143 [1.2651817401715446, 1.9941581887293254] |
| cubic | 65 | 17.86 | 11.34 | +6.52 | 0.0000 | True | +1.00 | +1.00 | 1.7617157282239244 [1.4650507156253678, 2.1029883511565197] |
| bbr_like | 65 | 17.86 | 17.85 | +0.01 | 0.0348 | False | +0.38 | +0.26 | 0.942261480936285 [0.8263053992718495, 1.0442039474806395] |
| vegas_like | 65 | 17.86 | 17.38 | +0.48 | 0.2948 | False | -0.23 | +0.08 | 1.0679038686121918 [0.9281407273951379, 1.2686742073981867] |
| random | 65 | 17.86 | 7.60 | +10.26 | 0.0000 |  | +0.88 | +0.97 | 2.0163807264840794 [1.5764913517182573, 2.60071487436268] |
| ml_only | 65 | 17.86 | 7.82 | +10.04 | 0.0000 |  | +0.23 | +0.70 | 1.9061390481051974 [1.2789705967465448, 2.9992185876781075] |
| rl_only | 65 | 17.86 | 17.36 | +0.50 | 0.3271 |  | -0.20 | +0.06 | 1.164593019630137 [0.9867987341385892, 1.4611661658924922] |
| rl_only_nodelay | 65 | 17.86 | 4.02 | +13.84 | 0.0000 |  | +0.85 | +0.92 | 6.326464939890632 [3.5067781937059577, 10.719701250281583] |
| rl_only_noloss | 65 | 17.86 | 17.97 | -0.11 | 0.9177 |  | -0.38 | -0.20 | 1.0470977644612898 [0.8791573711037347, 1.259840012201915] |
| rl_only_p95rtt | 65 | 101.89 | 143.25 | -41.35 | 0.0000 |  | -0.66 | -0.66 | None [None, None] |

## Guardrails (aggregate means; ok=False = violated)

- vs reno: m_mean_rtt_ms = 80.710 vs 107.612 -> OK
- vs reno: m_p95_rtt_ms = 101.894 vs 207.454 -> OK
- vs reno: m_loss_rate = 0.000 vs 0.021 -> OK
- vs reno: m_jain_fairness = 0.988 vs 0.991 -> OK
- vs cubic: m_mean_rtt_ms = 80.710 vs 112.470 -> OK
- vs cubic: m_p95_rtt_ms = 101.894 vs 171.528 -> OK
- vs cubic: m_loss_rate = 0.000 vs 0.031 -> OK
- vs cubic: m_jain_fairness = 0.988 vs 0.983 -> OK
- vs bbr_like: m_mean_rtt_ms = 80.710 vs 85.703 -> OK
- vs bbr_like: m_p95_rtt_ms = 101.894 vs 120.222 -> OK
- vs bbr_like: m_loss_rate = 0.000 vs 0.001 -> OK
- vs bbr_like: m_jain_fairness = 0.988 vs 0.997 -> OK
- vs vegas_like: m_mean_rtt_ms = 80.710 vs 96.474 -> OK
- vs vegas_like: m_p95_rtt_ms = 101.894 vs 120.445 -> OK
- vs vegas_like: m_loss_rate = 0.000 vs 0.002 -> OK
- vs vegas_like: m_jain_fairness = 0.988 vs 0.993 -> OK
