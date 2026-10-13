"""Double-DQN agent for congestion-window control (PyTorch, CPU).

Small MLP Q-network (2x128 ReLU). All hyperparameters are fixed a priori
in configs/learning.yaml and pre-registered; no test-set tuning.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.pos = 0
        self.size = 0
        rng = np.random.default_rng(seed)
        self._rng = rng
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.int64)
        self.r = np.zeros(self.capacity, dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(self.capacity, dtype=np.float32)

    def add(self, s, a, r, s2, done) -> None:
        p = self.pos
        self.s[p] = s
        self.a[p] = a
        self.r[p] = r
        self.s2[p] = s2
        self.d[p] = 1.0 if done else 0.0
        self.pos = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int) -> Tuple[np.ndarray, ...]:
        idx = self._rng.integers(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQNAgent:
    """Double DQN with Huber loss, gradient clipping, periodic target sync."""

    def __init__(self, state_dim: int, n_actions: int, cfg: Optional[dict] = None,
                 seed: int = 0, device: str = "cpu") -> None:
        cfg = cfg or {}
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        self.device = torch.device(device)
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lr = float(cfg.get("lr", 1e-3))
        self.batch = int(cfg.get("batch", 128))
        self.target_sync_steps = int(cfg.get("target_sync_steps", 1000))
        self.train_every = int(cfg.get("train_every", 2))
        self.warmup_steps = int(cfg.get("warmup_steps", 2000))
        self.grad_clip = float(cfg.get("grad_clip", 10.0))
        self.hidden = int(cfg.get("hidden", 128))

        self.online = QNetwork(state_dim, n_actions, self.hidden).to(self.device)
        self.target = QNetwork(state_dim, n_actions, self.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(int(cfg.get("buffer_size", 100_000)), state_dim, seed)
        self._env_steps = 0
        self._train_steps = 0
        self._last_train_step = 0
        self.rng = random.Random(self.seed)
        self.loss_history: list = []
        self.q_mean_history: list = []

    # --------------------------------------------------------------- acting
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            q = self.online(s)
            return int(q.argmax(dim=1).item())

    # -------------------------------------------------------------- learning
    def remember(self, s, a, r, s2, done) -> None:
        self.buffer.add(s, a, r, s2, done)
        self._env_steps += 1

    @property
    def env_steps(self) -> int:
        return self._env_steps

    @property
    def train_steps(self) -> int:
        return self._train_steps

    def maybe_train(self) -> Optional[float]:
        """Called once per env step; trains every train_every steps."""
        if self.buffer.size < max(self.batch, self.warmup_steps):
            return None
        if self._env_steps % self.train_every != 0:
            return None
        return self.train_step()

    def train_step(self) -> float:
        s, a, r, s2, d = self.buffer.sample(self.batch)
        s_t = torch.as_tensor(s, device=self.device)
        a_t = torch.as_tensor(a, device=self.device)
        r_t = torch.as_tensor(r, device=self.device)
        s2_t = torch.as_tensor(s2, device=self.device)
        d_t = torch.as_tensor(d, device=self.device)
        q_online = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # double DQN: online selects, target evaluates
            a2 = self.online(s2_t).argmax(dim=1)
            q2 = self.target(s2_t).gather(1, a2.unsqueeze(1)).squeeze(1)
            target = r_t + self.gamma * (1.0 - d_t) * q2
        loss = F.smooth_l1_loss(q_online, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.opt.step()
        self._train_steps += 1
        if self._train_steps % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.loss_history.append(float(loss.item()))
        self.q_mean_history.append(float(q_online.mean().item()))
        return float(loss.item())

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"DQN checkpoint not found: {path}")
        obj = torch.load(path, map_location=device, weights_only=True)
        self = cls(state_dim=obj["state_dim"], n_actions=obj["n_actions"],
                   cfg={"hidden": obj.get("hidden", 128)}, seed=obj.get("seed", 0),
                   device=device)
        self.online.load_state_dict(obj["online"])
        self.target.load_state_dict(obj["target"])
        return self

    def model_size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.online.parameters())
