"""Unit tests for the ML predictor and DQN agent."""
import numpy as np
import pytest

from src.models.ml_predictor import MLPredictor, TARGET_NAMES
from src.models.dqn import DQNAgent, ReplayBuffer


def _make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 18)).astype(np.float32)
    Y = {"qd_ms": X[:, 4] * 20 + rng.normal(0, 1, n),
         "loss_sig": np.clip(X[:, 10] * 0.02 + rng.normal(0, 0.01, n), 0, 1),
         "rate_mbps": X[:, 0] * 10 + rng.normal(0, 1, n)}
    return X, Y


def test_ml_fit_predict_shapes():
    X, Y = _make_data()
    pred = MLPredictor()
    pred.fit(X, Y)
    x = X[:5]
    out = pred.predict(x[0])
    assert out.shape == (3,)
    assert np.all(out >= 0)
    assert np.all(np.isfinite(out))
    batch = pred.predict_target("rate_mbps", x)
    assert batch.shape == (5,)


def test_ml_save_load_roundtrip(tmp_path):
    X, Y = _make_data()
    pred = MLPredictor()
    pred.fit(X, Y)
    path = tmp_path / "ml.joblib"
    pred.save(path)
    pred2 = MLPredictor.load(path)
    x = X[7]
    np.testing.assert_allclose(pred.predict(x), pred2.predict(x), rtol=1e-6)
    assert pred2.model_size_bytes() > 0


def test_ml_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MLPredictor.load(tmp_path / "nope.joblib")


def test_ml_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MLPredictor().predict(np.zeros(18, dtype=np.float32))


def test_dqn_shapes_and_act_bounds():
    agent = DQNAgent(18, 8, seed=0)
    s = np.zeros(18, dtype=np.float32)
    for _ in range(20):
        a = agent.act(s, epsilon=1.0)
        assert 0 <= a < 8
    a = agent.act(s, epsilon=0.0)
    assert 0 <= a < 8


def test_dqn_replay_and_train():
    agent = DQNAgent(18, 8, cfg={"warmup_steps": 0, "batch": 8, "n_step": 3,
                                 "train_every": 1}, seed=0)
    rng = np.random.default_rng(0)
    for i in range(40):
        s = rng.random(18).astype(np.float32)
        s2 = rng.random(18).astype(np.float32)
        agent.remember(s, 2, 0.5, s2, False, flow_id=1)
        agent.maybe_train()
    # flush with a terminal transition
    agent.remember(np.zeros(18, np.float32), 0, 0.1,
                   np.zeros(18, np.float32), True, flow_id=1)
    assert agent.train_steps > 0
    # n-step emission happened
    assert agent.buffer.size > 0


def test_dqn_nstep_per_flow_isolation():
    agent = DQNAgent(18, 8, cfg={"n_step": 3}, seed=0)
    rng = np.random.default_rng(0)
    for i in range(5):
        agent.remember(rng.random(18).astype(np.float32), 1, float(i),
                       rng.random(18).astype(np.float32), False, flow_id=7)
        agent.remember(rng.random(18).astype(np.float32), 1, float(i) + 100,
                       rng.random(18).astype(np.float32), False, flow_id=9)
    # sliding window emits on every step once the window is full:
    # 5 steps -> 3 emissions per flow
    assert agent.buffer.size == 6
    assert len(agent._pending) == 2


def test_dqn_save_load(tmp_path):
    agent = DQNAgent(18, 8, seed=0)
    s = np.random.default_rng(1).random(18).astype(np.float32)
    path = tmp_path / "dqn.pt"
    agent.save(path)
    agent2 = DQNAgent.load(path)
    assert agent.act(s, 0.0) == agent2.act(s, 0.0)
    assert agent2.model_size_bytes() > 0


def test_replay_buffer_wraparound():
    buf = ReplayBuffer(10, 3, seed=0)
    for i in range(25):
        buf.add(np.full(3, i, dtype=np.float32), 0, 0.0,
                np.full(3, i, dtype=np.float32), False)
    assert buf.size == 10
    assert buf.pos == 5
    s, a, r, s2, d, gn = buf.sample(4)
    assert s.shape == (4, 3)
    assert gn.shape == (4,)
    assert np.all(gn == 1.0)
