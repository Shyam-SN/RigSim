"""Fault injection engine orchestrating all 24 synthetic fault types."""

from typing import Any

import numpy as np

from rigsim.config.camera import CameraIntrinsics
from rigsim.config.faults import FaultLabel, FaultScenarioConfig
from rigsim.config.transform import Transform
from rigsim.config.types import FaultType
from rigsim.faults.base import BaseFaultInjector
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
from rigsim.sim.imu import IMUSample

INJECTOR_REGISTRY: dict[FaultType, type[BaseFaultInjector]] = {
    # Timing / Transport faults (1-12)
    FaultType.RANDOM_FRAME_DROP: RandomFrameDropInjector,
    FaultType.BURST_FRAME_DROP: BurstFrameDropInjector,
    FaultType.FPS_JITTER: FPSJitterInjector,
    FaultType.LOW_EFFECTIVE_FPS: LowEffectiveFPSInjector,
    FaultType.CLOCK_DRIFT: ClockDriftInjector,
    FaultType.CLOCK_OFFSET: ClockOffsetInjector,
    FaultType.IMU_CAM_OFFSET: IMUCamOffsetInjector,
    FaultType.CLOCK_STEP: ClockStepInjector,
    FaultType.TIMESTAMP_NON_MONOTONIC: TimestampSanityInjector,
    FaultType.TIMESTAMP_DUPLICATE: TimestampSanityInjector,
    FaultType.HOST_LATENCY_SPIKE: HostLatencySpikeInjector,
    FaultType.IMU_DROPOUT: IMUDropoutInjector,
    FaultType.STREAM_DISCONNECT: StreamDisconnectInjector,
    # Image Content faults (13-20)
    FaultType.MOTION_BLUR: MotionBlurInjector,
    FaultType.OVEREXPOSURE: ExposureFaultInjector,
    FaultType.UNDEREXPOSURE: ExposureFaultInjector,
    FaultType.LENS_OCCLUSION: LensOcclusionInjector,
    FaultType.FROZEN_FRAME: FrozenFrameInjector,
    FaultType.CORRUPTED_FRAME: CorruptedFrameInjector,
    FaultType.HEAVY_NOISE: HeavyNoiseInjector,
    FaultType.CAMERA_BUMP: CameraBumpInjector,
    FaultType.INTRINSIC_SHIFT: IntrinsicShiftInjector,
    # IMU Content faults (21-24)
    FaultType.IMU_SATURATION: IMUSaturationInjector,
    FaultType.IMU_BIAS_JUMP: IMUBiasJumpInjector,
    FaultType.IMU_HIGH_NOISE: IMUHighNoiseInjector,
    FaultType.IMU_AXIS_FLIP: IMUAxisFlipInjector,
}


class FaultEngine:
    """Instantiates, schedules, and applies configured synthetic faults across sensor streams."""

    def __init__(
        self,
        config: FaultScenarioConfig | None = None,
        epoch_base_ns: int = 1_700_000_000_000_000_000,
        seed: int = 42,
    ):
        self.config = config
        self.epoch_base_ns = epoch_base_ns
        self.seed = seed
        self.injectors: list[BaseFaultInjector] = []
        self._labels: list[FaultLabel] = []

        if config is not None:
            self._build_injectors(config)

    def _build_injectors(self, config: FaultScenarioConfig) -> None:
        """Instantiate injector classes from scenario config specs."""
        for idx, spec in enumerate(config.faults):
            cls = INJECTOR_REGISTRY.get(spec.fault_type)
            if cls is None:
                continue

            # Convert seconds to epoch nanoseconds
            t_start_s = spec.t_start_s if spec.t_start_s is not None else 0.0
            t_end_s = spec.t_end_s if spec.t_end_s is not None else 1000.0

            t_start_ns = self.epoch_base_ns + int(round(t_start_s * 1e9))
            t_end_ns = self.epoch_base_ns + int(round(t_end_s * 1e9))

            extra_args: dict[str, Any] = {**spec.params}
            if spec.fault_type == FaultType.OVEREXPOSURE:
                extra_args["mode"] = "overexposure"
            elif spec.fault_type == FaultType.UNDEREXPOSURE:
                extra_args["mode"] = "underexposure"
            elif spec.fault_type == FaultType.TIMESTAMP_NON_MONOTONIC:
                extra_args["mode"] = "non_monotonic"
            elif spec.fault_type == FaultType.TIMESTAMP_DUPLICATE:
                extra_args["mode"] = "duplicate"

            injector = cls(
                stream_id=spec.stream_id,
                t_start_ns=t_start_ns,
                t_end_ns=t_end_ns,
                severity=spec.severity,
                seed=self.seed + idx * 31,
                **extra_args,
            )
            self.injectors.append(injector)
            self._labels.append(injector.create_label())

    def should_drop_frame(self, stream_id: str, timestamp_ns: int) -> bool:
        """Check if any timing injector requests dropping this camera frame."""
        for inj in self.injectors:
            if inj.stream_id == stream_id:
                if isinstance(
                    inj,
                    (
                        RandomFrameDropInjector,
                        BurstFrameDropInjector,
                        LowEffectiveFPSInjector,
                        StreamDisconnectInjector,
                    ),
                ):
                    if inj.should_drop(timestamp_ns):
                        return True
        return False

    def should_drop_imu(self, stream_id: str, timestamp_ns: int) -> bool:
        """Check if IMU dropout or disconnect requests dropping this sample."""
        for inj in self.injectors:
            if inj.stream_id == stream_id:
                if isinstance(inj, (IMUDropoutInjector, StreamDisconnectInjector)):
                    if inj.should_drop(timestamp_ns):
                        return True
        return False

    def modify_device_time(self, stream_id: str, device_time_ns: int) -> int:
        """Apply timestamp modifications (drift, offset, step, jitter, sanity)."""
        ts = device_time_ns
        for inj in self.injectors:
            if inj.stream_id == stream_id:
                if isinstance(
                    inj,
                    (
                        ClockDriftInjector,
                        ClockOffsetInjector,
                        IMUCamOffsetInjector,
                        ClockStepInjector,
                        FPSJitterInjector,
                        TimestampSanityInjector,
                    ),
                ):
                    ts = inj.modify_timestamp(ts)
        return ts

    def modify_host_time(self, stream_id: str, host_time_ns: int, device_time_ns: int) -> int:
        """Apply host transport latency spikes."""
        h_ts = host_time_ns
        for inj in self.injectors:
            if inj.stream_id == stream_id and isinstance(inj, HostLatencySpikeInjector):
                h_ts = inj.modify_host_time(h_ts, device_time_ns)
        return h_ts

    def modify_image(self, stream_id: str, image_rgb: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Apply image content faults (blur, exposure, occlusion, noise, corruption, freeze)."""
        img = image_rgb
        for inj in self.injectors:
            if inj.stream_id == stream_id:
                if isinstance(
                    inj,
                    (
                        MotionBlurInjector,
                        ExposureFaultInjector,
                        LensOcclusionInjector,
                        FrozenFrameInjector,
                        CorruptedFrameInjector,
                        HeavyNoiseInjector,
                    ),
                ):
                    img = inj.modify_image(img, timestamp_ns)
        return img

    def modify_extrinsics(
        self, stream_id: str, transform: Transform, timestamp_ns: int
    ) -> Transform:
        """Apply camera bump / extrinsic shifts."""
        t = transform
        for inj in self.injectors:
            if inj.stream_id == stream_id and isinstance(inj, CameraBumpInjector):
                t = inj.modify_extrinsics(t, timestamp_ns)
        return t

    def modify_intrinsics(
        self, stream_id: str, intrinsics: CameraIntrinsics, timestamp_ns: int
    ) -> CameraIntrinsics:
        """Apply intrinsic / focal length shifts."""
        k = intrinsics
        for inj in self.injectors:
            if inj.stream_id == stream_id and isinstance(inj, IntrinsicShiftInjector):
                k = inj.modify_intrinsics(k, timestamp_ns)
        return k

    def modify_imu_sample(self, stream_id: str, sample: IMUSample) -> IMUSample:
        """Apply IMU content faults (saturation, bias jump, noise, axis flip)."""
        s = sample
        for inj in self.injectors:
            if inj.stream_id == stream_id:
                if isinstance(
                    inj,
                    (
                        IMUSaturationInjector,
                        IMUBiasJumpInjector,
                        IMUHighNoiseInjector,
                        IMUAxisFlipInjector,
                    ),
                ):
                    s = inj.modify_sample(s)
        return s

    def get_ground_truth_labels(self) -> list[FaultLabel]:
        """Return all ground-truth fault labels for evaluation."""
        return self._labels.copy()
