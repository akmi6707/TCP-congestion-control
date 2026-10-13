import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest


@pytest.fixture
def tiny_scenario():
    """A 2-second scenario for fast tests."""
    return {
        "name": "tiny", "duration_s": 2.0, "bandwidth_mbps": 4.0,
        "base_rtt_ms": 20.0, "queue_bdp_factor": 1.0, "loss_rate": 0.0,
        "n_flows": 1, "warmup_s": 0.5, "seed": 0,
    }


@pytest.fixture
def lossy_scenario():
    return {
        "name": "lossy", "duration_s": 3.0, "bandwidth_mbps": 4.0,
        "base_rtt_ms": 20.0, "queue_bdp_factor": 1.0, "loss_rate": 0.02,
        "n_flows": 1, "warmup_s": 0.5, "seed": 3,
    }
