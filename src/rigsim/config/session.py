"""Stream and Session manifests."""

from typing import Any

from pydantic import BaseModel, Field

from rigsim.config.camera import CameraDistortion, CameraIntrinsics
from rigsim.config.transform import Transform
from rigsim.config.types import SensorKind, SessionSource


class Stream(BaseModel):
    """Metadata describing one logical sensor data stream."""

    stream_id: str
    kind: SensorKind
    nominal_rate_hz: float = Field(..., gt=0.0)
    resolution: tuple[int, int] | None = None
    encoding: str = "jpeg"
    clock_id: str
    frame_id: str
    intrinsics: CameraIntrinsics | None = None
    distortion: CameraDistortion | None = None
    extrinsics: Transform


class Session(BaseModel):
    """Root session manifest representing a single recording."""

    session_id: str
    source: SessionSource
    rig_config_name: str
    created_at: str  # ISO 8601 UTC
    operator: str
    notes: str | None = None
    streams: list[Stream] = Field(default_factory=list)
    mcap_path: str
    ground_truth_labels_path: str | None = None
    ground_truth_trajectory_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
