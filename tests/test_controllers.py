"""Unit tests for controllers: bounds, action space, learning variants."""
import math

import numpy as np
import pytest

from src.controllers.learning import (_ActionSpace, DEFAULT_LEARNING_CFG,
                                       N_ACTIONS, make_learning_controller_factory)
from src.controllers import make_controller_factory
from src.environment.sim import simulate


def test_action_space_bounds():
    space = _ActionSpace(DEFAULT_LEARNING_CFG)

    class F:
        cwnd = 10.0

    f = F()
    for a in range(N_ACTIONS):
        f.cwnd = 10.0
        mult = space.apply(f, a, bdp_est_pkts=50.0)
        assert 1.0 <= f.cwnd <= 4096.0
        assert math.isfinite(f.cwnd)
        assert mult > 0
    # extreme: cwnd at cap stays bounded
    f.cwnd = 1e9
    space.apply(f, 5, 50.0)
    assert f.cwnd <= 4096.0
    f.cwnd = 0.0001
    space.apply(f, 4, 50.0)
    assert f.cwnd >= 1.0


def test_set_bdp_actions_anchored():
    space = _ActionSpace(DEFAULT_LEARNING_CFG)

    class F:
        cwnd = 1000.0

    f = F()
    space.apply(f, 6, bdp_est_pkts=80.0)
    assert f.cwnd == pytest.approx(80.0)
    space.apply(f, 7, bdp_est_pkts=80.0)
    assert f.cwnd == pytest.approx(120.0)


def test_all_conventional_controllers_execute(tiny_scenario):
    for name in ("reno", "cubic", "bbr_like", "vegas_like"):
        res = simulate(tiny_scenario, make_controller_factory(name))
        assert res.per_flow[0]["rcv_inorder_pkts"] > 0, name


def test_random_learning_controller_runs(tiny_scenario):
    factory = make_learning_controller_factory("random", seed=1)
    res = simulate(tiny_scenario, factory)
    assert res.per_flow[0]["rcv_inorder_pkts"] > 0
    stats = factory.inference_stats()
    assert stats["n_steps"] > 0


def test_ml_only_controller_runs(tiny_scenario, tmp_path):
    from src.models.ml_predictor import MLPredictor
    # train a tiny predictor on synthetic rows so ml_only can be built
    rng = np.random.default_rng(0)
    X = rng.random((50, 18)).astype(np.float32)
    Y = {"qd_ms": rng.random(50), "loss_sig": rng.random(50) * 0.01,
         "rate_mbps": rng.random(50) * 10}
    pred = MLPredictor()
    pred.fit(X, Y)
    path = tmp_path / "ml.joblib"
    pred.save(path)
    factory = make_learning_controller_factory("ml_only", ml_path=str(path))
    res = simulate(tiny_scenario, factory)
    assert res.per_flow[0]["rcv_inorder_pkts"] > 0


def test_hybrid_requires_ml_model():
    with pytest.raises(ValueError):
        make_learning_controller_factory("hybrid")
    with pytest.raises(ValueError):
        make_learning_controller_factory("rl_only")
    with pytest.raises(ValueError):
        make_learning_controller_factory("no_such_controller")


def test_hybrid_wires_policy_act_fn(tiny_scenario, tmp_path):
    """Regression: the hybrid factory must connect the DQN policy to the
    controller's act_fn (an earlier version left act_fn None, silently
    falling back to random actions)."""
    from src.models.dqn import DQNAgent
    from src.models.ml_predictor import MLPredictor
    import numpy as np
    # tiny ML model
    rng = np.random.default_rng(0)
    X = rng.random((30, 18)).astype(np.float32)
    Y = {"qd_ms": rng.random(30), "loss_sig": rng.random(30) * 0.01,
         "rate_mbps": rng.random(30) * 10}
    pred = MLPredictor()
    pred.fit(X, Y)
    ml_path = tmp_path / "ml.joblib"
    pred.save(ml_path)
    dqn_path = tmp_path / "dqn.pt"
    DQNAgent(21, N_ACTIONS, seed=0).save(dqn_path)

    factory = make_learning_controller_factory(
        "hybrid", policy_path=str(dqn_path), ml_path=str(ml_path))
    ctrl = factory.build_fn(0)
    assert ctrl.act_fn is not None, "hybrid controller must use the DQN policy"
    a = ctrl.act_fn(np.zeros(21, dtype=np.float32))
    assert 0 <= a < N_ACTIONS
    # and the factory-built controller runs in the simulator
    res = simulate(tiny_scenario, factory)
    assert res.per_flow[0]["rcv_inorder_pkts"] > 0


def test_rl_only_with_random_policy_runs(tiny_scenario, tmp_path):
    from src.models.dqn import DQNAgent
    agent = DQNAgent(18, N_ACTIONS, seed=0)
    path = tmp_path / "dqn.pt"
    agent.save(path)
    factory = make_learning_controller_factory("rl_only", policy_path=str(path))
    res = simulate(tiny_scenario, factory)
    assert res.per_flow[0]["rcv_inorder_pkts"] > 0
