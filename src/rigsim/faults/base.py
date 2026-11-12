"""Base class and common types for RigSim fault injectors."""

from abc import ABC
from typing import Any

import numpy as np

from rigsim.config.faults import FaultLabel
from rigsim.config.types import FaultType


class BaseFaultInjector(ABC):
    """Abstract base class for all synthetic fault injectors.

    Every injector is deterministic given a seed, operates on specific streams/windows,
    and returns ground-truth FaultLabel objects accurately describing the modification.
    """

    def __init__(
        self,
        fault_type: FaultType,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        self.fault_type = fault_type
        self.stream_id = stream_id
        self.t_start_ns = t_start_ns
        self.t_end_ns = t_end_ns
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.seed = seed
        self.params: dict[str, Any] = kwargs
        self.rng = np.random.default_rng(seed)

    def is_active(self, timestamp_ns: int) -> bool:
        """Check if injector is active at the given timestamp."""
        return self.t_start_ns <= timestamp_ns <= self.t_end_ns

    def create_label(self, **extra_params: Any) -> FaultLabel:
        """Generate ground-truth label for evaluation."""
        merged_params = {**self.params, **extra_params}
        return FaultLabel(
            fault_type=self.fault_type,
            stream_id=self.stream_id,
            t_start_ns=self.t_start_ns,
            t_end_ns=self.t_end_ns,
            severity=self.severity,
            params=merged_params,
        )
