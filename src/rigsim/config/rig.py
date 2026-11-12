"""Rig specification schema."""

from pydantic import BaseModel, Field

from rigsim.config.camera import CameraSpec, ExtrinsicsSpec
from rigsim.config.clock import ClockModel, SyncGroupSpec
from rigsim.config.imu import IMUSpec


class SyncStrobeSpec(BaseModel):
    """Sync strobe specification for visual inter-camera sync."""

    enabled: bool = True
    position: tuple[float, float, float] = (0.0, 0.0, 0.75)
    pattern_type: str = "prbs"  # pseudo-random binary sequence
    baud_rate_hz: float = Field(10.0, gt=0.0)


class CharucoBoardSpec(BaseModel):
    """ChArUco board target in the scene."""

    squares_x: int = Field(7, gt=2)
    squares_y: int = Field(5, gt=2)
    square_length_m: float = Field(0.04, gt=0.0)
    marker_length_m: float = Field(0.03, gt=0.0)
    dictionary: str = "DICT_5X5_100"
    pose: ExtrinsicsSpec


class SceneFeaturesSpec(BaseModel):
    """Visual scene targets and strobe emitters."""

    sync_strobe: SyncStrobeSpec | None = None
    charuco_board: CharucoBoardSpec | None = None


class RigConfig(BaseModel):
    """Complete multi-sensor rig definition."""

    schema_version: str = "1.0.0"
    rig_name: str
    description: str
    sync_groups: list[SyncGroupSpec] = Field(default_factory=list)
    clocks: list[ClockModel] = Field(default_factory=list)
    imu: IMUSpec | None = None
    cameras: list[CameraSpec] = Field(default_factory=list)
    scene_features: SceneFeaturesSpec | None = None

    def get_camera(self, stream_id: str) -> CameraSpec | None:
        for cam in self.cameras:
            if cam.stream_id == stream_id:
                return cam
        return None

    def get_clock(self, clock_id: str) -> ClockModel | None:
        for clock in self.clocks:
            if clock.clock_id == clock_id:
                return clock
        return None
