"""Integration tests: end-to-end run pipeline, suites, failure recording."""
import json

import pytest

from src.evaluation.runner import run_one
from src.evaluation.suite import run_suite, suite_experiment_id
from src.metrics import compute_metrics


def test_run_one_conventional(tiny_scenario, tmp_path):
    rec = run_one(tiny_scenario, "reno", seed=0, experiment="it-test",
                  results_dir=tmp_path)
    assert rec["status"] == "ok"
    assert rec["metrics"]["goodput_mbps"] > 0
    assert (tmp_path / "runs" / f"{rec['run_id']}.json").is_file()
    assert "reno" in (tmp_path / "index.csv").read_text()


def test_run_one_failure_recorded(tiny_scenario, tmp_path):
    rec = run_one(tiny_scenario, "does_not_exist", seed=0, experiment="it-fail",
                  results_dir=tmp_path)
    assert rec["status"] == "failed"
    assert "unknown controller" in rec["error"]
    assert (tmp_path / "runs" / f"{rec['run_id']}.json").is_file()


def test_run_one_learning_controller(tiny_scenario, tmp_path):
    from src.controllers.learning import make_learning_controller_factory
    factory = make_learning_controller_factory("random", seed=0)
    from src.environment.sim import simulate
    res = simulate(tiny_scenario, factory)
    m = compute_metrics(res)
    assert m["goodput_mbps"] >= 0
    # inference timing was collected
    assert factory.inference_stats()["n_steps"] > 0


def test_suite_runs_and_persists(tiny_scenario, tmp_path):
    scenarios = [tiny_scenario, dict(tiny_scenario, bandwidth_mbps=8.0, name="tiny2")]
    experiment = suite_experiment_id("ittest")
    records = run_suite(scenarios,
                        {"reno": {}, "vegas_like": {}},
                        seeds=[0, 1], experiment=experiment, suite="ittest",
                        results_dir=tmp_path, n_workers=2,
                        save_series_controllers=["reno"])
    assert len(records) == 8  # 2 x 2 x 2
    assert all(r["status"] == "ok" for r in records)
    n_series = sum(1 for r in records if r.get("series_path"))
    assert n_series == 4  # reno runs saved series
    # loading the run records back
    loaded = tmp_path / "runs"
    n_files = len(list(loaded.glob("*.json")))
    assert n_files == 8


def test_analyze_experiment_on_tiny_suite(tiny_scenario, tmp_path):
    from src.evaluation.analyze import analyze_experiment, write_analysis
    experiment = suite_experiment_id("ittest2")
    run_suite([tiny_scenario], {"reno": {}, "vegas_like": {}}, seeds=[0, 1, 2],
              experiment=experiment, suite="ittest2", results_dir=tmp_path,
              n_workers=1)
    analysis = analyze_experiment(tmp_path, experiment,
                                  primary_controller="vegas_like",
                                  baselines=["reno"],
                                  learning=["vegas_like"])
    assert analysis["n_ok_runs"] == 6
    assert "reno" in analysis["primary_tests"]
    assert analysis["primary_tests"]["reno"]["n_pairs"] == 3
    out = write_analysis(tmp_path, analysis, "ittest2")
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["experiment"] == experiment


def test_failed_model_path_records_failure(tiny_scenario, tmp_path):
    rec = run_one(tiny_scenario, "rl_only", seed=0, experiment="it-mlfail",
                  results_dir=tmp_path, policy_path="models/nope.pt")
    assert rec["status"] == "failed"
    assert "nope.pt" in rec["error"]
