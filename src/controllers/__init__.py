"""Controller registry.

make_controller_factory(name, **kwargs) returns a callable
``factory(flow_id) -> BaseController`` so that multi-flow scenarios
instantiate one controller per flow (sharing model/policy objects for the
learning controllers).
"""
from __future__ import annotations

from .base import BaseController
from .reno import RenoController
from .cubic import CubicController
from .bbr_like import BBRLikeController
from .vegas_like import VegasLikeController

CONVENTIONAL = {
    "reno": RenoController,
    "cubic": CubicController,
    "bbr_like": BBRLikeController,
    "vegas_like": VegasLikeController,
}


def make_controller_factory(name: str, **kwargs):
    if name in CONVENTIONAL:
        cls = CONVENTIONAL[name]
        return lambda flow_id: cls(flow_id=flow_id)
    # learning-based controllers are built by their training/eval modules
    # (they need a loaded model); dispatch here so callers have one entry point.
    if name in ("rl_only", "hybrid", "ml_only", "random"):
        from .learning import make_learning_controller_factory
        return make_learning_controller_factory(name, **kwargs)
    raise ValueError(f"unknown controller: {name!r}")


__all__ = [
    "BaseController", "RenoController", "CubicController", "BBRLikeController",
    "VegasLikeController", "CONVENTIONAL", "make_controller_factory",
]
