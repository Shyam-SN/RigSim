"""Clock models, synchronization groups, and transport latency models."""

from pydantic import BaseModel, Field

from rigsim.config.types import SyncMode


class ClockStepEvent(BaseModel):
    """Sudden clock jump event (e.g. NTP step, unsynced host jump)."""

    t_sim_ns: int = Field(..., description="Simulation timestamp when step occurs")
    step_ns: int = Field(..., description="Offset delta injected at step point")


class ClockModel(BaseModel):
    """Realistic sensor clock model.

    t_device = (1 + drift_ppm * 1e-6) * t_sim + offset_ns + N(0, jitter_std_ns) + step_events
    """

    clock_id: str
    offset_ns: int = Field(0, description="Initial constant clock offset in ns")
    drift_ppm: float = Field(0.0, description="Clock frequency error in parts per million")
    jitter_std_ns: float = Field(
        0.0, ge=0.0, description="Timestamp jitter standard deviation in ns"
    )
    step_events: list[ClockStepEvent] = Field(default_factory=list)


class SyncGroupSpec(BaseModel):
    """Definition of a hardware or software synchronization group."""

    group_id: str
    sync_mode: SyncMode
    stream_ids: list[str]


class TransportLatencyModel(BaseModel):
    """Host transport latency model: log_time_ns = device_time_ns + latency."""

    mean_ms: float = Field(5.0, ge=0.0)
    jitter_std_ms: float = Field(1.5, ge=0.0)
    spike_prob: float = Field(
        0.005, ge=0.0, le=1.0, description="Probability of transport latency spike"
    )
    spike_multiplier: float = Field(5.0, ge=1.0)
