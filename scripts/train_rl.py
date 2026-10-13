#!/usr/bin/env python3
"""Train a DQN congestion-control policy.

Variants:
  rl_only        : 18-dim state, pre-registered reward
  hybrid         : 21-dim state (18 + 3 ML predictions), same reward
  rl_only_nodelay: ablation, w_queue_delay = 0
  rl_only_noloss : ablation, w_loss = 0

The scenario per episode is sampled from configs/curriculum.yaml. Every
flow's controller shares the policy and the replay buffer (centralized
training, decentralized execution). Periodic validation runs (epsilon=0)
select the best checkpoint; the final model and the training log are saved.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from src.controllers.learning import (make_learning_controller_factory,
                                       N_ACTIONS)
from src.environment.features import (N_BASE_FEATURES, N_HYBRID_FEATURES)
from src.environment.sim import simulate
from src.metrics import compute_metrics
from src.models.dqn import DQNAgent
from src.models.ml_predictor import MLPredictor
from src.training.curriculum import sample_scenario
from src.utils import io as uio
from src.utils.config import load_yaml

log = logging.getLogger("train_rl")


def train(variant: str, args) -> int:
    cur = load_yaml(args.curriculum)
    learn_cfg = load_yaml(args.learning)
    n_episodes = args.episodes if args.episodes else int(cur["n_episodes"])
    seed = args.seed

    state_dim = N_BASE_FEATURES + (N_HYBRID_FEATURES if variant == "hybrid" else 0)
    reward_weights = None
    if variant == "rl_only_nodelay":
        reward_weights = dict(learn_cfg["reward_weights"]); reward_weights["w_queue_delay"] = 0.0
        base_variant = "rl_only"
    elif variant == "rl_only_noloss":
        reward_weights = dict(learn_cfg["reward_weights"]); reward_weights["w_loss"] = 0.0
        base_variant = "rl_only"
    else:
        base_variant = variant

    ml_model = None
    if variant == "hybrid":
        ml_model = MLPredictor.load(args.ml_path)

    agent = DQNAgent(state_dim, N_ACTIONS, cfg=learn_cfg.get("dqn"), seed=seed)
    dqn_cfg = cur.get("dqn", {})
    eps_start = float(dqn_cfg.get("epsilon_start", 1.0))
    eps_end = float(dqn_cfg.get("epsilon_end", 0.05))
    eps_decay = float(dqn_cfg.get("epsilon_decay_steps", 250_000))

    best_val = -float("inf")
    train_log = {"variant": variant, "seed": seed, "state_dim": state_dim,
                 "n_episodes": n_episodes, "episodes": [], "code_version": uio.git_commit()}
    t_start = time.time()

    def make_factories(eps: float, training: bool):
        def act_fn(state, _eps=eps):
            return agent.act(state, _eps)

        def train_fn(s, a, r, s2, done):
            agent.remember(s, a, r, s2, done)
            agent.maybe_train()

        factory = make_learning_controller_factory(
            base_variant, cfg=learn_cfg,
            policy_agent=agent,
            ml_model=ml_model,
            train_fn=train_fn if training else None,
            reward_weights=reward_weights, seed=seed)
        return factory

    val_every = int(cur.get("validation", {}).get("every_episodes", 40))
    val_scenarios = cur.get("validation", {}).get("scenarios", [])

    for ep in range(n_episodes):
        # epsilon by env step count
        frac = min(1.0, agent.env_steps / max(eps_decay, 1))
        eps_now = eps_start + (eps_end - eps_start) * frac
        sc = sample_scenario(cur, ep, seed)
        factory = make_factories(eps_now, training=True)
        t0 = time.perf_counter()
        try:
            res = simulate(sc, factory, seed=ep)
        except Exception as exc:  # noqa: BLE001
            log.error("episode %d failed: %s", ep, exc)
            train_log["episodes"].append({"ep": ep, "error": str(exc)})
            continue
        ep_wall = time.perf_counter() - t0
        # close the final transition of each controller
        for c in factory.instances:
            if hasattr(c, "finish_episode"):
                c.finish_episode(None)
        # episode statistics
        rewards = [c.prev_reward for c in factory.instances if hasattr(c, "prev_reward")]
        m = compute_metrics(res)
        train_log["episodes"].append({
            "ep": ep, "goodput_mbps": m["goodput_mbps"],
            "bandwidth_mbps": sc["bandwidth_mbps"], "n_flows": sc["n_flows"],
            "loss_rate_cfg": sc["loss_rate"], "mean_reward": float(np.mean(rewards)) if rewards else None,
            "eps": eps_now, "env_steps": agent.env_steps,
            "train_steps": agent.train_steps, "wall_s": ep_wall,
        })
        if (ep + 1) % 10 == 0:
            last = train_log["episodes"][-1]
            log.info("ep %4d gp=%6.2f/%5.1f rew=%6.3f eps=%.3f steps=%7d t=%5.1fs "
                     "loss=%.4f q=%.2f",
                     ep + 1, last["goodput_mbps"], last["bandwidth_mbps"],
                     last["mean_reward"] or 0, eps_now, agent.env_steps,
                     ep_wall,
                     float(np.mean(agent.loss_history[-50:])) if agent.loss_history else 0.0,
                     float(np.mean(agent.q_mean_history[-50:])) if agent.q_mean_history else 0.0)

        if (ep + 1) % val_every == 0:
            scores = []
            for vsc in val_scenarios:
                f = make_factories(0.0, training=False)
                try:
                    r = simulate(vsc, f, seed=seed)
                    scores.append(compute_metrics(r)["goodput_mbps"])
                except Exception as exc:  # noqa: BLE001
                    log.error("validation run failed: %s", exc)
            val = float(np.mean(scores)) if scores else float("nan")
            train_log["episodes"][-1]["val_goodput_mbps"] = val
            log.info("  -> validation mean goodput: %.3f Mbps (best %.3f)", val, best_val)
            if np.isfinite(val) and val > best_val:
                best_val = val
                agent.save(args.out)
                train_log["best_val"] = best_val
                train_log["best_episode"] = ep + 1
                log.info("  -> saved best checkpoint to %s", args.out)

    # final checkpoint under a separate name (best is the one used for eval)
    agent.save(args.out.replace(".pt", "_final.pt"))
    train_log["total_time_s"] = time.time() - t_start
    train_log["final_env_steps"] = agent.env_steps
    train_log["final_train_steps"] = agent.train_steps
    os.makedirs(os.path.dirname(os.path.abspath(args.log_out)), exist_ok=True)
    with open(args.log_out, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=1)
    log.info("done: best val %.3f (ep %s), total %.1f min, log %s",
             best_val, train_log.get("best_episode"), train_log["total_time_s"] / 60,
             args.log_out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["rl_only", "hybrid", "rl_only_nodelay", "rl_only_noloss"])
    ap.add_argument("--curriculum", default="configs/curriculum.yaml")
    ap.add_argument("--learning", default="configs/learning.yaml")
    ap.add_argument("--ml-path", default="models/ml_predictor.joblib")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-out", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if args.out is None:
        args.out = f"models/dqn_{args.variant}.pt"
    if args.log_out is None:
        args.log_out = f"results/training/dqn_{args.variant}_log.json"
    return train(args.variant, args)


if __name__ == "__main__":
    raise SystemExit(main())
