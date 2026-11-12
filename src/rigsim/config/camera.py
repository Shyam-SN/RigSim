"""Camera specs, intrinsics, and distortion models."""

from pydantic import BaseModel, Field

from rigsim.config.transform import Transform
from rigsim.config.types import DistortionModel, SensorKind


class CameraIntrinsics(BaseModel):
    """Pinhole camera intrinsic parameters."""

    fx: float = Field(..., gt=0.0, description="Focal length x (pixels)")
    fy: float = Field(..., gt=0.0, description="Focal length y (pixels)")
    cx: float = Field(..., gt=0.0, description="Principal point x (pixels)")
    cy: float = Field(..., gt=0.0, description="Principal point y (pixels)")
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    fov_deg: float | None = Field(None, gt=0.0, description="Field of view (degrees)")


class CameraDistortion(BaseModel):
    """Lens distortion parameters."""

    model: DistortionModel = DistortionModel.RADTAN
    coeffs: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0],
        description="Distortion coefficients: [k1, k2, p1, p2, (k3)]",
    )


class ExtrinsicsSpec(BaseModel):
    """Extrinsics specification in YAML configs (convenient representation)."""

    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def to_transform(self) -> Transform:
        return Transform.from_translation_quaternion(self.translation, self.rotation_xyzw)


class CameraSpec(BaseModel):
    """Specification of a camera in a rig configuration file."""

    stream_id: str
    kind: SensorKind = SensorKind.RGB
    nominal_rate_hz: float = Field(30.0, gt=0.0)
    resolution: tuple[int, int] = Field((1280, 720), description="(width, height)")
    fov_deg: float = Field(75.0, gt=0.0, lt=180.0)
    clock_id: str
    frame_id: str
    exposure_time_ms: float = Field(10.0, gt=0.0)
    mounted_on_body: str | None = Field(
        None, description="MuJoCo body name if attached to moving component (e.g. wrist)"
    )
    distortion: CameraDistortion = Field(default_factory=lambda: CameraDistortion())
    extrinsics_to_rig: ExtrinsicsSpec
