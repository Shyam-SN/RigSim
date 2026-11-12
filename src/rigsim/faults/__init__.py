"""Fault injection framework for RigSim."""

from rigsim.faults.base import BaseFaultInjector
from rigsim.faults.engine import INJECTOR_REGISTRY, FaultEngine
from rigsim.faults.image import (
    CameraBumpInjector,
    CorruptedFrameInjector,
    ExposureFaultInjector,
    FrozenFrameInjector,
    HeavyNoiseInjector,
    IntrinsicShiftInjector,
    LensOcclusionInjector,
    MotionBlurInjector,
)
from rigsim.faults.imu import (
    IMUAxisFlipInjector,
    IMUBiasJumpInjector,
    IMUHighNoiseInjector,
    IMUSaturationInjector,
)
from rigsim.faults.timing import (
    BurstFrameDropInjector,
    ClockDriftInjector,
    ClockOffsetInjector,
    ClockStepInjector,
    FPSJitterInjector,
    HostLatencySpikeInjector,
    IMUCamOffsetInjector,
    IMUDropoutInjector,
    LowEffectiveFPSInjector,
    RandomFrameDropInjector,
    StreamDisconnectInjector,
    TimestampSanityInjector,
)

__all__ = [
    "BaseFaultInjector",
    "FaultEngine",
    "INJECTOR_REGISTRY",
    "RandomFrameDropInjector",
    "BurstFrameDropInjector",
    "FPSJitterInjector",
    "LowEffectiveFPSInjector",
    "ClockDriftInjector",
    "ClockOffsetInjector",
    "IMUCamOffsetInjector",
    "ClockStepInjector",
    "TimestampSanityInjector",
    "HostLatencySpikeInjector",
    "IMUDropoutInjector",
    "StreamDisconnectInjector",
    "MotionBlurInjector",
    "ExposureFaultInjector",
    "LensOcclusionInjector",
    "FrozenFrameInjector",
    "CorruptedFrameInjector",
    "HeavyNoiseInjector",
    "CameraBumpInjector",
    "IntrinsicShiftInjector",
    "IMUSaturationInjector",
    "IMUBiasJumpInjector",
    "IMUHighNoiseInjector",
    "IMUAxisFlipInjector",
]
