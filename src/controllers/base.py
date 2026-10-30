
from __future__ import annotations


class BaseController:
    name = "base"

    def __init__(self, flow_id: int = 0) -> None:
        self.flow_id = flow_id

    def on_new_acks(self, flow, n_new: int) -> None:
        """n_new previously-unacked packets were cumulatively acked."""

    def on_fast_retransmit(self, flow) -> None:
        """3 duplicate ACKs detected a loss (fast retransmit happened)."""

    def on_rto(self, flow) -> None:
        """Retransmission timeout fired."""

    def on_tick(self, flow, obs: dict) -> None:
        """Per simulator tick; obs carries sender-visible observations."""

    def controller_info(self) -> dict:
        """Metadata persisted with the run (model ids, sizes...)."""
        return {"name": self.name}
