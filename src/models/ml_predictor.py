"""Supervised ML component: next-interval network-state prediction.

Given the RL-only feature vector available at a control point (i.e., built
from the interval that just closed), three HistGradientBoostingRegressor
models predict the ABOUT-TO-RUN interval's:

  1. qd_ms      - mean queueing delay (ms)
  2. loss_sig   - sender-observable loss signal fraction (see features.loss_signal)
  3. rate_mbps  - delivered goodput (Mbps)

All targets are sender-observable, so the predictor's semantics transfer to
a real stack. Models are fixed a priori (max_iter=300, lr=0.08, default
regularization) - no test-set-informed tuning.

Training data comes from logged episodes of the conventional baselines and
a uniform-random-action controller across the training scenario
distribution (see training/collect_ml_data.py), with scenario-disjoint
train/validation splits.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..environment.features import FEATURE_NAMES

TARGET_NAMES: List[str] = ["qd_ms", "loss_sig", "rate_mbps"]


class MLPredictor:
    VERSION = "1.0"

    def __init__(self) -> None:
        self.models: Dict[str, object] = {}
        self.train_info: Dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, Y: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Fit the three models. Returns training metrics."""
        from sklearn.ensemble import HistGradientBoostingRegressor
        metrics = {}
        for name in TARGET_NAMES:
            y = np.asarray(Y[name], dtype=np.float64)
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.08, random_state=0)
            model.fit(X, y)
            self.models[name] = model
            pred = model.predict(X)
            # in-sample metrics (validation metrics are computed by the
            # caller on the scenario-disjoint split)
            metrics[name] = {
                "mae": float(np.mean(np.abs(pred - y))),
                "r2": float(1.0 - np.sum((pred - y) ** 2) /
                            max(np.sum((y - y.mean()) ** 2), 1e-12)),
            }
        self.train_info = {"n_train": int(X.shape[0]), "in_sample": metrics}
        return metrics

    def evaluate(self, X: np.ndarray, Y: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        metrics = {}
        for name in TARGET_NAMES:
            y = np.asarray(Y[name], dtype=np.float64)
            pred = self.predict_target(name, X)
            ss_res = float(np.sum((pred - y) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            metrics[name] = {
                "mae": float(np.mean(np.abs(pred - y))),
                "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
                "mean_target": float(y.mean()) if len(y) else 0.0,
                "mean_pred": float(pred.mean()) if len(pred) else 0.0,
            }
        return metrics

    # -------------------------------------------------------------- predict
    def predict_target(self, name: str, X: np.ndarray) -> np.ndarray:
        model = self.models.get(name)
        if model is None:
            raise RuntimeError(f"ML model '{name}' not fitted/loaded")
        out = model.predict(X)
        if name == "loss_sig":
            out = np.clip(out, 0.0, 1.0)
        elif name == "qd_ms":
            out = np.clip(out, 0.0, 5000.0)
        elif name == "rate_mbps":
            out = np.clip(out, 0.0, 1e4)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Single-sample prediction -> [qd_ms, loss_sig, rate_mbps]."""
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        return np.array([
            self.predict_target("qd_ms", x)[0],
            self.predict_target("loss_sig", x)[0],
            self.predict_target("rate_mbps", x)[0],
        ], dtype=np.float32)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "version": self.VERSION,
                "models": self.models,
                "train_info": self.train_info,
                "feature_names": FEATURE_NAMES,
                "target_names": TARGET_NAMES,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "MLPredictor":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ML predictor checkpoint not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self = cls()
        self.models = obj["models"]
        self.train_info = obj.get("train_info", {})
        missing = [t for t in TARGET_NAMES if t not in self.models]
        if missing:
            raise RuntimeError(f"checkpoint missing targets: {missing}")
        return self

    def model_size_bytes(self) -> int:
        try:
            return len(pickle.dumps(self.models))
        except Exception:
            return -1
