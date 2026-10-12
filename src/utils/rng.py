"""Deterministic RNG helpers.

All stochasticity in the simulator is derived from explicit (seed, kind)
streams so that runs are reproducible and so that different controllers in
the same scenario see the *same sequence* of link-loss dice rolls (best-effort
pairing for paired statistical tests; the exact packets they apply to still
depends on controller behavior, which is documented).
"""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np


def make_rng(seed: int, kind: str) -> np.random.Generator:
    """Create a numpy Generator deterministically derived from (seed, kind).

    kind is hashed with CRC32 (stable across interpreter runs and versions,
    unlike Python's builtin hash()).
    """
    kind_id = zlib.crc32(kind.encode("utf-8")) & 0xFFFFFFFF
    ss = np.random.SeedSequence(entropy=[int(seed) & 0xFFFFFFFF, int(kind_id)])
    return np.random.default_rng(ss)


def loss_count(n: int, p: float, u: float) -> int:
    """Deterministic 'binomial-like' loss count for a packet batch.

    Given a shared uniform draw ``u`` in [0,1) for a simulation tick, the
    number of lost packets out of ``n`` with per-packet loss probability ``p``
    is the integer part of n*p plus one extra packet if the fractional part
    exceeds ``u``.  Expectation is exactly n*p for uniform u, the mapping is
    deterministic in (n, p, u), and identical ``u`` sequences are used for all
    controllers in a scenario, which makes runs as comparable as the dynamics
    allow.
    """
    if p <= 0.0 or n <= 0:
        return 0
    expected = n * p
    k = int(expected)
    frac = expected - k
    if frac > 0.0 and u < frac:
        k += 1
    return min(k, n)


def sample_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Uniform sample in [lo, hi]."""
    return float(rng.uniform(lo, hi))


def choice(rng: np.random.Generator, options: Sequence):
    return options[int(rng.integers(0, len(options)))]
