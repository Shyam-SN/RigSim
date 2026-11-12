"""Fault definitions, injectors configuration, and ground-truth labels."""

from typing import Any

from pydantic import BaseModel, Field

from rigsim.config.types import FaultType


class FaultLabel(BaseModel):
    """Ground truth label for an injected or identified fault."""

    fault_type: FaultType
    stream_id: str
    t_start_ns: int
    t_end_ns: int
    severity: float = Field(..., ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class FaultInjectionSpec(BaseModel):
    """Specification of an injected fault in a scenario config."""

    fault_type: FaultType
    stream_id: str
    t_start_s: float | None = None
    t_end_s: float | None = None
    severity: float = Field(1.0, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class FaultScenarioConfig(BaseModel):
    """Scenario configuration defining a combination of faults."""

    schema_version: str = "1.0.0"
    scenario_name: str
    description: str
    faults: list[FaultInjectionSpec] = Field(default_factory=list)
