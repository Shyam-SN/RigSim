"""QA results, verdicts, and threshold configuration."""

from typing import Any

from pydantic import BaseModel, Field

from rigsim.config.types import QAVerdict


class TimeWindow(BaseModel):
    """Time interval in nanoseconds."""

    t_start_ns: int
    t_end_ns: int


class QAResult(BaseModel):
    """Result of an individual QA check."""

    check_id: str  # e.g. T1_FRAME_DROPS, I1_BLUR, C2_CAMERA_BUMP
    stream_id: str
    verdict: QAVerdict
    metric_name: str
    value: float
    threshold: float
    flagged_windows: list[TimeWindow] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str


class SessionVerdict(BaseModel):
    """Overall session QA assessment and yield."""

    session_id: str
    overall_verdict: QAVerdict
    usable_time_ranges: list[TimeWindow] = Field(default_factory=list)
    usable_duration_s: float
    total_duration_s: float
    yield_ratio: float = Field(..., ge=0.0, le=1.0)
    summary_rules: list[str] = Field(default_factory=list)
    results: list[QAResult] = Field(default_factory=list)

    @property
    def usable_percentage(self) -> float:
        return self.yield_ratio * 100.0

    @property
    def usable_ranges(self) -> list[TimeWindow]:
        return self.usable_time_ranges

    @property
    def summary_sentence(self) -> str:
        return (
            "; ".join(self.summary_rules) if self.summary_rules else "All critical checks passed."
        )

    @property
    def critical_failures_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == QAVerdict.FAIL)

    @property
    def warnings_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == QAVerdict.WARN)


class TimingThresholds(BaseModel):
    max_drop_ratio: float = 0.01  # 1%
    max_drop_burst_frames: int = 2
    max_jitter_ms: float = 2.0
    min_effective_fps_ratio: float = 0.95
    max_inter_cam_offset_ms: float = 2.0
    max_inter_cam_drift_ppm: float = 10.0
    max_imu_cam_offset_ms: float = 5.0
    max_transport_latency_spike_ms: float = 25.0


class ImageThresholds(BaseModel):
    min_normalized_blur_ratio: float = 0.50
    min_mean_brightness: float = 20.0
    max_mean_brightness: float = 235.0
    max_clipping_ratio: float = 0.15
    max_frozen_frame_count: int = 2
    max_corrupted_frame_count: int = 0
    occlusion_min_uniform_ratio: float = 0.60


class CalibrationThresholds(BaseModel):
    max_charuco_reprojection_rms_px: float = 0.50
    max_stereo_reprojection_rms_px: float = 0.70
    max_bump_rotation_deg: float = 0.50
    max_bump_translation_mm: float = 3.0
    max_epipolar_error_px: float = 1.0


class IMUThresholds(BaseModel):
    max_gravity_error_m_s2: float = 0.20  # |norm - 9.81|
    max_gyro_static_bias_rad_s: float = 0.01
    max_accel_static_bias_m_s2: float = 0.10
    max_saturation_samples: int = 0


class QAThresholdsConfig(BaseModel):
    """Global QA thresholds configuration."""

    schema_version: str = "1.0.0"
    timing: TimingThresholds = Field(default_factory=lambda: TimingThresholds())
    image: ImageThresholds = Field(default_factory=lambda: ImageThresholds())
    calibration: CalibrationThresholds = Field(default_factory=lambda: CalibrationThresholds())
    imu: IMUThresholds = Field(default_factory=lambda: IMUThresholds())
    critical_checks: list[str] = Field(
        default_factory=lambda: [
            "T1_FRAME_DROPS",
            "T4_TIMESTAMP_SANITY",
            "T6_INTER_CAM_SYNC",
            "T7_IMU_CAM_OFFSET",
            "I3_OCCLUSION",
            "I4_FROZEN_FRAMES",
            "I5_CORRUPTION",
            "C2_CAMERA_BUMP",
            "C3_STEREO_CONSISTENCY",
            "M1_IMU_GRAVITY",
        ]
    )
