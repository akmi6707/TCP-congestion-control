"""Instrumented controller wrapper for ML data collection.

Wraps any conventional controller, delegating all control decisions to it,
while independently running the SAME observation/feature pipeline the
learning controllers use. At each control boundary it logs a training row:
(feature vector at time t) -> (outcomes of interval t).

The conventional controller's per-interval cwnd change is mapped to the
nearest multiplicative action index (and the corresponding reward is
computed), so features 16 (prev_action) and 17 (prev_reward) carry the
same semantics as in the learning-controller data. Documented approximation.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

import numpy as np

from ..controllers.base import BaseController
from ..controllers.learning import DEFAULT_LEARNING_CFG, N_ACTIONS
from ..environment.features import (IntervalAggregator, build_features)
from ..models.reward import compute_reward


class InstrumentedController(BaseController):
    def __init__(self, flow_id: int, inner: BaseController,
                 interval_ticks: int = 5, dt_ms: float = 5.0,
                 collect_fn: Optional[Callable] = None) -> None:
        super().__init__(flow_id)
        self.inner = inner
        self.aggr = IntervalAggregator(interval_ticks, dt_ms)
        self.collect_fn = collect_fn
        self._pending_features: Optional[np.ndarray] = None
        self._prev_cwnd = None
        self._prev_action = 2
        self._prev_reward = 0.0
        # multiplicative action factors for pseudo-action mapping
        self._mult_actions = [(i, a["factor"]) for i, a in
                               enumerate(DEFAULT_LEARNING_CFG["actions"])
                               if a["type"] == "mult"]

    # ------------------------------------------------------- delegate hooks
    def on_new_acks(self, flow, n_new: int) -> None:
        self.inner.on_new_acks(flow, n_new)

    def on_fast_retransmit(self, flow) -> None:
        self.inner.on_fast_retransmit(flow)

    def on_rto(self, flow) -> None:
        self.inner.on_rto(flow)

    # ------------------------------------------------------------ main tick
    def on_tick(self, flow, obs: dict) -> None:
        # let the real controller act first
        cwnd_before = flow.cwnd
        self.inner.on_tick(flow, obs)
        self.aggr.on_tick(obs)
        if not self.aggr.interval_ready():
            return
        prev_stats = self.aggr.prev_stats
        stats = self.aggr.take_stats()

        # pseudo-action: nearest multiplicative action to the realized change
        mult = 1.0
        if self._prev_cwnd is not None and self._prev_cwnd > 0:
            mult = flow.cwnd / self._prev_cwnd
        best_i, best_d = 2, float("inf")
        for i, f in self._mult_actions:
            d = abs(math.log2(f) - math.log2(max(mult, 1e-9)))
            if d < best_d:
                best_i, best_d = i, d
        action_idx = best_i

        # complete the pending row: features_t -> outcomes of interval t
        if self.collect_fn is not None and self._pending_features is not None:
            self.collect_fn(self._pending_features, stats, flow, self)

        # reward for the interval that just closed (under the pseudo-action)
        r = compute_reward(stats, mult)
        self._prev_reward = r

        f = build_features(stats, prev_stats, action_idx, r, N_ACTIONS)
        self._pending_features = f  # waits for next interval's outcomes
        self._prev_cwnd = flow.cwnd
        self._prev_action = action_idx

    def controller_info(self) -> dict:
        info = {"name": f"instrumented_{self.inner.name}"}
        return info
