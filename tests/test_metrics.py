"""Unit tests for metrics (Jain, aggregation, analysis helpers)."""
import numpy as np
import pytest

from src.metrics.metrics import jain
from src.evaluation.analyze import (cliffs_delta, holm_correction,
                                    paired_test, geometric_mean_ratios)
from src.utils.rng import loss_count


def test_jain_perfect_and_degenerate():
    assert jain([5, 5, 5, 5]) == pytest.approx(1.0)
    assert jain([10, 0]) == pytest.approx(0.5)
    assert jain([0, 0]) == 0.0
    assert np.isnan(jain([]))


def test_jain_two_flows():
    # (a+b)^2 / (2*(a^2+b^2))
    v = jain([3, 1])
    assert v == pytest.approx(16 / 20)


def test_loss_count_deterministic_expectation():
    # E[k] over uniform u must equal n*p
    total = 0
    n_trials = 2000
    for i in range(n_trials):
        total += loss_count(100, 0.01, (i * 0.0005) % 1.0)
    assert abs(total / n_trials - 1.0) < 0.2
    assert loss_count(0, 0.5, 0.99) == 0
    assert loss_count(10, 0.0, 0.99) == 0
    assert loss_count(10, 1.0, 0.99) == 10


def test_cliffs_delta():
    assert cliffs_delta([10, 10], [1, 1]) == 1.0
    assert cliffs_delta([1, 1], [10, 10]) == -1.0
    assert cliffs_delta([5, 5], [5, 5]) == 0.0
    assert np.isnan(cliffs_delta([], []))


def test_holm_correction():
    # Holm steps through sorted p-values with thresholds alpha/(m-rank);
    # it STOPS at the first non-rejection: [0.01, 0.04, 0.03] -> 0.01<=0.0167
    # rejects; 0.03 > 0.025 fails -> stop (0.04 never tested).
    rej = holm_correction([0.01, 0.04, 0.03])
    assert rej == [True, False, False]
    # all small -> all reject
    assert holm_correction([0.001, 0.005, 0.01]) == [True, True, True]
    assert holm_correction([0.9, 0.95]) == [False, False]


def test_paired_test_on_synthetic_data():
    import pandas as pd
    a = pd.DataFrame({"scenario": list("aabbc"), "seed": [0, 1, 0, 1, 0],
                      "m": [10, 11, 12, 13, 14]})
    b = pd.DataFrame({"scenario": list("aabbc"), "seed": [0, 1, 0, 1, 0],
                      "m": [8, 9, 10, 11, 12]})
    out = paired_test(a, b, "m", alternative="greater")
    assert out["n_pairs"] == 5
    assert out["mean_diff"] == pytest.approx(2.0)
    assert out["cliffs_delta"] == 1.0
    assert out["wilcoxon_p"] < 0.05


def test_geometric_mean_ratios():
    import pandas as pd
    a = pd.DataFrame({"scenario": list("abc"), "m": [10.0, 20.0, 40.0]})
    b = pd.DataFrame({"scenario": list("abc"), "m": [10.0, 10.0, 10.0]})
    out = geometric_mean_ratios(a, b, "m", n_boot=100)
    assert out["geo_mean_ratio"] == pytest.approx((1 * 2 * 4) ** (1 / 3))
    assert out["ci95_lo"] <= out["geo_mean_ratio"] <= out["ci95_hi"]
