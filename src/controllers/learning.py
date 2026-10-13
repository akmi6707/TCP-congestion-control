"""Learning-based controllers: random, RL-only (DQN), hybrid (ML+RL), ML-only.

Action space (pre-registered, bounded by construction):
  0: cwnd x 0.5        (multiplicative decrease)
  1: cwnd x 0.8        (mild decrease)
  2: cwnd x 1.0        (hold)
  3: cwnd x 1.02       (gentle growth)
  4: cwnd x 1.1        (growth)
  5: cwnd x 1.3        (aggressive growth)
  6: cwnd = 1.0 x BDP estimate   (re-anchor to estimated pipe size)
  7: cwnd = 1.5 x BDP estimate  (re-anchor high, e.g. after regime change)

cwnd is clipped to [CWND_MIN, CWND_MAX] packets after every action, so no
unbounded or invalid output is possible. BDP estimate uses the sender-side
delivered-rate estimate times min-RTT (floor 4 packets).

Bootstrap: until the feature vector is valid (first RTT sample + first
delivered data), the controller ramps additively (+2 packets per control
interval) and applies a 0.7 multiplicative backoff on loss events. This is
part of the controller (identical for all learning variants), not the
transport.

All learning controllers act on 25 ms control intervals (5 sim ticks) and
share the same observation/feature pipeline; the ONLY differences are:
  - random:  actions sampled uniformly (data-collection diversity + floor)
  - rl_only: state = 18 base features, actions from the DQN policy
  - hybrid:  state = 18 base features + 3 ML predictions (next-interval
             queueing delay, loss signal, rate), actions from the DQN policy
  - ml_only: no RL; sets cwnd from the ML predictions directly
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from .base import BaseController
from ..environment.features import (IntervalAggregator, IntervalStats,
                                    build_features, append_hybrid_predictions,
                                    N_BASE_FEATURES)
from ..models.reward import compute_reward, DEFAULT_WEIGHTS
from ..utils.rng import make_rng

DEFAULT_LEARNING_CFG = {
    "dt_ms": 5.0,
    "interval_ticks": 5,              # 25 ms control interval
    "cwnd_min": 1.0,
    "cwnd_max": 4096.0,
    "actions": [
        {"type": "mult", "factor": 0.5},
        {"type": "mult", "factor": 0.8},
        {"type": "mult", "factor": 1.0},
        {"type": "mult", "factor": 1.02},
        {"type": "mult", "factor": 1.1},
        {"type": "mult", "factor": 1.3},
        {"type": "set_bdp", "factor": 1.0},
        {"type": "set_bdp", "factor": 1.5},
    ],
    "bootstrap": {"add_pkts": 2.0, "loss_backoff": 0.7},
}

N_ACTIONS = len(DEFAULT_LEARNING_CFG["actions"])


def _merge_cfg(cfg: Optional[dict]) -> dict:
    out = dict(DEFAULT_LEARNING_CFG)
    if cfg:
        out.update(cfg)
    return out


class _ActionSpace:
    def __init__(self, cfg: dict) -> None:
        self.actions = cfg["actions"]
        self.cwnd_min = float(cfg["cwnd_min"])
        self.cwnd_max = float(cfg["cwnd_max"])

    def apply(self, flow, action_idx: int, bdp_est_pkts: float) -> float:
        """Apply action; returns the effective multiplier (for the
        oscillation penalty; set_bdp actions report new/old ratio)."""
        act = self.actions[action_idx]
        old = max(flow.cwnd, 1.0)
        if act["type"] == "mult":
            new = flow.cwnd * float(act["factor"])
        else:  # set_bdp
            bdp = max(bdp_est_pkts, 4.0)
            new = bdp * float(act["factor"])
        flow.cwnd = float(min(max(new, self.cwnd_min), self.cwnd_max))
        return flow.cwnd / old


class LearningController(BaseController):
    """Shared pipeline for random / rl_only / hybrid controllers."""

    def __init__(self, flow_id: int, cfg: Optional[dict] = None,
                 act_fn: Optional[Callable[[np.ndarray], int]] = None,
                 ml_predictor=None, collect_fn: Optional[Callable] = None,
                 train_fn: Optional[Callable] = None,
                 reward_weights: Optional[Dict[str, float]] = None,
                 seed: int = 0) -> None:
        super().__init__(flow_id)
        self.cfg = _merge_cfg(cfg)
        self.space = _ActionSpace(self.cfg)
        self.aggr = IntervalAggregator(self.cfg["interval_ticks"], self.cfg["dt_ms"])
        self.act_fn = act_fn
        self.ml_predictor = ml_predictor
        self.collect_fn = collect_fn
        self.train_fn = train_fn
        self.reward_weights = reward_weights or DEFAULT_WEIGHTS
        self.prev_action = 2
        self.prev_mult = 1.0
        self.prev_reward = 0.0
        self.rng = make_rng(seed, f"learning_ctrl_{flow_id}")
        self.inference_ms: List[float] = []
        # pending ML row: features at time t waiting for interval t outcomes
        self._pending_features: Optional[np.ndarray] = None
        # pending DQN transition pieces
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action_taken: Optional[int] = None
        self.episode_done = False

    # ---------------------------------------------------------------- hooks
    def on_new_acks(self, flow, n_new: int) -> None:
        pass  # learning controllers act at control-interval granularity

    def on_rto(self, flow) -> None:
        # transport already collapsed cwnd; controller sees it in features
        pass

    def on_tick(self, flow, obs: dict) -> None:
        self.aggr.on_tick(obs)
        if not self.aggr.interval_ready():
            return
        prev_stats = self.aggr.prev_stats
        stats = self.aggr.take_stats()

        # 1) reward for the action that governed the interval just closed
        r = compute_reward(stats, self.prev_mult, self.reward_weights)
        self.prev_reward = r

        # 2) features available NOW (built from the closed interval)
        f = build_features(stats, prev_stats, self.prev_action,
                           self.prev_reward, N_ACTIONS)
        if f is None:
            # bootstrap mode: safe additive ramp with loss backoff
            flow.cwnd = min(flow.cwnd + self.cfg["bootstrap"]["add_pkts"],
                            self.space.cwnd_max)
            if stats.loss_events > 0 or stats.rto_events > 0:
                flow.cwnd = max(flow.cwnd * self.cfg["bootstrap"]["loss_backoff"],
                                self.space.cwnd_min)
            self._pending_features = None
            self._prev_state = None
            self._prev_action_taken = None
            return

        # 3) complete pending ML data row: (features_t -> outcomes_t)
        if self.collect_fn is not None and self._pending_features is not None:
            self.collect_fn(self._pending_features, stats, flow, self)

        # 4) build state (hybrid appends ML predictions)
        t0 = time.perf_counter()
        if self.ml_predictor is not None:
            preds = self.ml_predictor.predict(f)
            state = append_hybrid_predictions(f, preds[0], preds[1], preds[2])
        else:
            state = f
        if self.act_fn is not None:
            action = int(self.act_fn(state))
        else:
            action = self.rng.integers(0, N_ACTIONS)
        self.inference_ms.append((time.perf_counter() - t0) * 1000.0)

        # 5) store DQN transition for the PREVIOUS control point
        if self.train_fn is not None and self._prev_state is not None:
            self.train_fn(self._prev_state, self._prev_action_taken, r, state,
                          False, self.flow_id)

        # 6) apply the action for the upcoming interval
        mult = self.space.apply(flow, action, stats.bdp_est_pkts)
        self.prev_action = action
        self.prev_mult = mult

        # 7) remember for next control point
        self._pending_features = f
        self._prev_state = state
        self._prev_action_taken = action

    def finish_episode(self, flow) -> None:
        """Called by the training loop at episode end to close the last
        transition with done=True."""
        if self.train_fn is not None and self._prev_state is not None:
            self.train_fn(self._prev_state, self._prev_action_taken,
                          self.prev_reward, self._prev_state, True, self.flow_id)
        self._prev_state = None
        self._prev_action_taken = None
        self._pending_features = None

    def controller_info(self) -> dict:
        info = {"name": self.name, "n_actions": N_ACTIONS,
                "interval_ticks": self.cfg["interval_ticks"]}
        if self.ml_predictor is not None:
            info["ml_model_size_bytes"] = self.ml_predictor.model_size_bytes()
        return info


class RandomController(LearningController):
    name = "random"


class RLController(LearningController):
    name = "rl_only"


class HybridController(LearningController):
    name = "hybrid"


class MLOnlyController(BaseController):
    """Supervised-ML-only controller (no RL): steers cwnd toward the ML
    predicted achievable rate x min-RTT with 15% headroom; backs off when
    predicted loss signal exceeds 1%. Moves 25% toward the target each
    control interval (pre-registered)."""
    name = "ml_only"

    def __init__(self, flow_id: int, ml_predictor, cfg: Optional[dict] = None,
                 seed: int = 0) -> None:
        super().__init__(flow_id)
        self.cfg = _merge_cfg(cfg)
        self.ml = ml_predictor
        self.aggr = IntervalAggregator(self.cfg["interval_ticks"], self.cfg["dt_ms"])
        self.inference_ms: List[float] = []
        self.headroom = 1.15
        self.move_frac = 0.25
        self.loss_backoff_threshold = 0.01

    def on_tick(self, flow, obs: dict) -> None:
        self.aggr.on_tick(obs)
        if not self.aggr.interval_ready():
            return
        stats = self.aggr.take_stats()
        f = build_features(stats, self.aggr.prev_stats, 0, 0.0, N_ACTIONS)
        if f is None:
            flow.cwnd = min(flow.cwnd + self.cfg["bootstrap"]["add_pkts"],
                            self.cfg["cwnd_max"])
            if stats.loss_events > 0 or stats.rto_events > 0:
                flow.cwnd = max(flow.cwnd * self.cfg["bootstrap"]["loss_backoff"],
                                self.cfg["cwnd_min"])
            return
        t0 = time.perf_counter()
        qd, loss, rate = self.ml.predict(f)
        self.inference_ms.append((time.perf_counter() - t0) * 1000.0)
        min_rtt = stats.min_rtt_end_ms or 50.0
        target = max(rate, 0.0) * 1e6 / 8.0 / 1460.0 * (min_rtt / 1000.0)
        target *= self.headroom
        if loss > self.loss_backoff_threshold:
            target = min(target, 0.9 * flow.cwnd)
        new = flow.cwnd + self.move_frac * (target - flow.cwnd)
        flow.cwnd = float(min(max(new, self.cfg["cwnd_min"]), self.cfg["cwnd_max"]))

    def on_rto(self, flow) -> None:
        pass

    def controller_info(self) -> dict:
        return {"name": self.name,
                "ml_model_size_bytes": self.ml.model_size_bytes()}


class ControllerFactory:
    """Creates one controller per flow while sharing the policy/model.
    Keeps references so runners can read inference stats afterwards."""

    def __init__(self, build_fn) -> None:
        self.build_fn = build_fn
        self.instances: List[BaseController] = []

    def __call__(self, flow_id: int) -> BaseController:
        c = self.build_fn(flow_id)
        self.instances.append(c)
        return c

    def inference_stats(self) -> dict:
        all_ms: List[float] = []
        for c in self.instances:
            ms = getattr(c, "inference_ms", None)
            if ms:
                all_ms.extend(ms)
        if not all_ms:
            return {"n_steps": 0}
        arr = np.asarray(all_ms)
        return {"n_steps": int(arr.size), "mean_ms": float(arr.mean()),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(arr.max())}

    def controller_info(self) -> dict:
        infos = [c.controller_info() for c in self.instances]
        return {"n_controllers": len(infos),
                "first": infos[0] if infos else {}}


def make_learning_controller_factory(name: str, cfg: Optional[dict] = None,
                                     policy_path: Optional[str] = None,
                                     policy_agent=None,
                                     ml_path: Optional[str] = None,
                                     ml_model=None,
                                     epsilon: float = 0.0, seed: int = 0,
                                     collect_fn: Optional[Callable] = None,
                                     train_fn: Optional[Callable] = None,
                                     reward_weights: Optional[Dict[str, float]] = None):
    """Build a ControllerFactory for a learning-based controller variant.

    policy_agent: an in-memory DQNAgent (training loop); policy_path: a
    saved checkpoint (evaluation). ml_model: in-memory MLPredictor; ml_path:
    saved predictor. Only one of each pair is needed.
    """
    merged = _merge_cfg(cfg)
    policy_act = None
    if name == "hybrid":
        if ml_model is None:
            from ..models.ml_predictor import MLPredictor
            if ml_path is None:
                raise ValueError("hybrid controller requires ml_path or ml_model")
            ml_model = MLPredictor.load(ml_path)
    elif name == "ml_only":
        if ml_model is None:
            from ..models.ml_predictor import MLPredictor
            if ml_path is None:
                raise ValueError("ml_only controller requires ml_path or ml_model")
            ml_model = MLPredictor.load(ml_path)
        def build(flow_id, _ml=ml_model):
            return MLOnlyController(flow_id, _ml, merged, seed=seed + flow_id)
        return ControllerFactory(build)
    elif name in ("rl_only", "random"):
        if name == "rl_only":
            agent = policy_agent
            if agent is None:
                if policy_path is None:
                    raise ValueError("rl_only requires policy_agent or policy_path")
                from ..models.dqn import DQNAgent
                agent = DQNAgent.load(policy_path)
            agent.rng.seed(seed)

            def policy_act(state, _ag=agent, _eps=epsilon):
                return _ag.act(state, _eps)
    else:
        raise ValueError(f"unknown learning controller: {name!r}")

    cls = {"random": RandomController, "rl_only": RLController,
           "hybrid": HybridController}[name]

    def build(flow_id):
        return cls(
            flow_id=flow_id, cfg=merged, act_fn=policy_act,
            ml_predictor=ml_model, collect_fn=collect_fn, train_fn=train_fn,
            reward_weights=reward_weights, seed=seed + flow_id)

    return ControllerFactory(build)
