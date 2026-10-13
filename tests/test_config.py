"""Unit tests for configuration validation and IO persistence."""
import json

import pytest

from src.utils.config import validate_scenario, load_scenario, ConfigError
from src.utils import io as uio


def test_valid_scenario_passes():
    sc = validate_scenario({"name": "x", "duration_s": 10, "bandwidth_mbps": 5,
                            "base_rtt_ms": 40})
    assert sc["dt_ms"] == 5.0
    assert sc["n_flows"] == 1


def test_missing_keys_rejected():
    with pytest.raises(ConfigError):
        validate_scenario({"duration_s": 10})
    with pytest.raises(ConfigError):
        validate_scenario({"name": "x"})


def test_rtt_must_be_tick_multiple():
    with pytest.raises(ConfigError):
        validate_scenario({"name": "x", "duration_s": 10, "bandwidth_mbps": 5,
                           "base_rtt_ms": 37})
    # 40 ms is a multiple of 2*dt (10 ms)
    validate_scenario({"name": "x", "duration_s": 10, "bandwidth_mbps": 5,
                       "base_rtt_ms": 40})


def test_loss_rate_bounds():
    with pytest.raises(ConfigError):
        validate_scenario({"name": "x", "duration_s": 10, "bandwidth_mbps": 5,
                           "base_rtt_ms": 40, "loss_rate": 0.5})


def test_load_scenario_file(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("name: y\nduration_s: 10\nbandwidth_mbps: 4\nbase_rtt_ms: 20\nwarmup_s: 1.0\n")
    sc = load_scenario(p)
    assert sc["name"] == "y"
    with pytest.raises(ConfigError):
        load_scenario(tmp_path / "missing.yaml")


def test_run_record_never_overwrites(tmp_path):
    rec = {"run_id": "run-test", "status": "ok", "experiment": "t",
           "controller": "reno", "seed": 0, "scenario": {"name": "x"},
           "metrics": {"goodput_mbps": 1.0}}
    uio.write_run_record(rec, results_dir=tmp_path)
    with pytest.raises(FileExistsError):
        uio.write_run_record(rec, results_dir=tmp_path)
    # index has one row + header
    idx = (tmp_path / "index.csv").read_text()
    assert "run-test" in idx


def test_series_never_overwrites(tmp_path):
    uio.write_time_series("run-s", {"a": [1]}, results_dir=tmp_path)
    with pytest.raises(FileExistsError):
        uio.write_time_series("run-s", {"a": [2]}, results_dir=tmp_path)
