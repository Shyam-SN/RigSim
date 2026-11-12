"""RigSim configuration and data schemas."""

from rigsim.config.camera import CameraDistortion, CameraIntrinsics, CameraSpec, ExtrinsicsSpec
from rigsim.config.clock import ClockModel, ClockStepEvent, SyncGroupSpec, TransportLatencyModel
from rigsim.config.cost import StoragePricingConfig
from rigsim.config.episode import Episode
from rigsim.config.faults import FaultInjectionSpec, FaultLabel, FaultScenarioConfig
from rigsim.config.imu import IMUSaturation, IMUSpec
from rigsim.config.qa import (
    CalibrationThresholds,
    ImageThresholds,
    IMUThresholds,
    QAResult,
    QAThresholdsConfig,
    SessionVerdict,
    TimeWindow,
    TimingThresholds,
)
from rigsim.config.rig import CharucoBoardSpec, RigConfig, SceneFeaturesSpec, SyncStrobeSpec
from rigsim.config.session import Session, Stream
from rigsim.config.transform import Transform
from rigsim.config.types import (
    DistortionModel,
    FaultType,
    QAVerdict,
    SensorKind,
    SessionSource,
    SyncMode,
)

__all__ = [
    "SensorKind",
    "DistortionModel",
    "SyncMode",
    "SessionSource",
    "FaultType",
    "QAVerdict",
    "Transform",
    "CameraIntrinsics",
    "CameraDistortion",
    "ExtrinsicsSpec",
    "CameraSpec",
    "IMUSaturation",
    "IMUSpec",
    "ClockStepEvent",
    "ClockModel",
    "SyncGroupSpec",
    "TransportLatencyModel",
    "SyncStrobeSpec",
    "CharucoBoardSpec",
    "SceneFeaturesSpec",
    "RigConfig",
    "FaultLabel",
    "FaultInjectionSpec",
    "FaultScenarioConfig",
    "TimeWindow",
    "QAResult",
    "SessionVerdict",
    "TimingThresholds",
    "ImageThresholds",
    "CalibrationThresholds",
    "IMUThresholds",
    "QAThresholdsConfig",
    "Stream",
    "Session",
    "Episode",
    "StoragePricingConfig",
]
