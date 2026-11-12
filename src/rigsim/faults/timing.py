"""Timing and transport fault injectors with ground-truth label generation."""

from typing import Any

from rigsim.config.types import FaultType
from rigsim.faults.base import BaseFaultInjector


class RandomFrameDropInjector(BaseFaultInjector):
    """Fault 1: Randomly drop frames with configured drop rate."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 0.05,
        seed: int = 42,
        drop_probability: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.RANDOM_FRAME_DROP,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            drop_probability=drop_probability if drop_probability is not None else severity,
            **kwargs,
        )
        self.drop_prob = self.params["drop_probability"]

    def should_drop(self, timestamp_ns: int) -> bool:
        if not self.is_active(timestamp_ns):
            return False
        return bool(self.rng.uniform(0.0, 1.0) < self.drop_prob)


class BurstFrameDropInjector(BaseFaultInjector):
    """Fault 2: Drop a burst of N consecutive frames."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        burst_size: int = 5,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.BURST_FRAME_DROP,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            burst_size=burst_size,
            **kwargs,
        )
        self.burst_size = burst_size
        self.dropped_so_far = 0
        self.burst_triggered = False

    def should_drop(self, timestamp_ns: int) -> bool:
        if not self.is_active(timestamp_ns):
            return False
        if self.dropped_so_far < self.burst_size:
            self.dropped_so_far += 1
            return True
        return False


class FPSJitterInjector(BaseFaultInjector):
    """Fault 3: Inject abnormal timestamp jitter beyond nominal specs."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        jitter_std_ms: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.FPS_JITTER,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            jitter_std_ms=jitter_std_ms,
            **kwargs,
        )
        self.jitter_std_ns = jitter_std_ms * 1_000_000 * severity

    def modify_timestamp(self, timestamp_ns: int) -> int:
        if not self.is_active(timestamp_ns):
            return timestamp_ns
        jitter = int(self.rng.normal(0.0, self.jitter_std_ns))
        return max(0, timestamp_ns + jitter)


class LowEffectiveFPSInjector(BaseFaultInjector):
    """Fault 4: Effective FPS drops below nominal (thermal throttling simulation)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 0.5,
        seed: int = 42,
        target_fps_ratio: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.LOW_EFFECTIVE_FPS,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            target_fps_ratio=target_fps_ratio,
            **kwargs,
        )
        self.target_fps_ratio = target_fps_ratio
        self.counter = 0

    def should_drop(self, timestamp_ns: int) -> bool:
        if not self.is_active(timestamp_ns):
            return False
        self.counter += 1
        # Keep every 1/ratio frames
        keep_interval = int(round(1.0 / max(0.05, self.target_fps_ratio)))
        return (self.counter % keep_interval) != 0


class ClockDriftInjector(BaseFaultInjector):
    """Fault 5: Frequency drift between camera/sensor groups."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        drift_ppm: float = 50.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.CLOCK_DRIFT,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            drift_ppm=drift_ppm,
            **kwargs,
        )
        self.drift_ppm = drift_ppm * severity

    def modify_timestamp(self, timestamp_ns: int) -> int:
        if timestamp_ns < self.t_start_ns:
            return timestamp_ns
        dt = timestamp_ns - self.t_start_ns
        drift_delta = int(round(dt * (self.drift_ppm * 1e-6)))
        return timestamp_ns + drift_delta


class ClockOffsetInjector(BaseFaultInjector):
    """Fault 6: Constant time offset between sensor groups."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        offset_ms: float = 20.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.CLOCK_OFFSET,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            offset_ms=offset_ms,
            **kwargs,
        )
        self.offset_ns = int(round(offset_ms * 1_000_000 * severity))

    def modify_timestamp(self, timestamp_ns: int) -> int:
        if not self.is_active(timestamp_ns):
            return timestamp_ns
        return timestamp_ns + self.offset_ns


class IMUCamOffsetInjector(BaseFaultInjector):
    """Fault 7: Temporal offset between camera frames and IMU samples."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        offset_ms: float = 15.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.IMU_CAM_OFFSET,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            offset_ms=offset_ms,
            **kwargs,
        )
        self.offset_ns = int(round(offset_ms * 1_000_000 * severity))

    def modify_timestamp(self, timestamp_ns: int) -> int:
        if not self.is_active(timestamp_ns):
            return timestamp_ns
        return timestamp_ns + self.offset_ns


class ClockStepInjector(BaseFaultInjector):
    """Fault 8: Sudden step jump / clock discontinuity."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        step_ms: float = 100.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.CLOCK_STEP,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            step_ms=step_ms,
            **kwargs,
        )
        self.step_ns = int(round(step_ms * 1_000_000 * severity))

    def modify_timestamp(self, timestamp_ns: int) -> int:
        if timestamp_ns >= self.t_start_ns:
            return timestamp_ns + self.step_ns
        return timestamp_ns


class TimestampSanityInjector(BaseFaultInjector):
    """Fault 9: Non-monotonic or duplicate timestamps."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        mode: str = "non_monotonic",  # non_monotonic or duplicate
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=(
                FaultType.TIMESTAMP_NON_MONOTONIC
                if mode == "non_monotonic"
                else FaultType.TIMESTAMP_DUPLICATE
            ),
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            mode=mode,
            **kwargs,
        )
        self.mode = mode
        self.last_ts: int | None = None
        self.injected = False

    def modify_timestamp(self, timestamp_ns: int) -> int:
        if not self.is_active(timestamp_ns):
            self.last_ts = timestamp_ns
            return timestamp_ns

        # If first frame in window, record it as baseline
        if self.last_ts is None:
            self.last_ts = timestamp_ns
            return timestamp_ns

        if not self.injected:
            self.injected = True
            if self.mode == "duplicate":
                return self.last_ts
            else:
                # Step backward by 20 ms (non-monotonic)
                return self.last_ts - 20_000_000

        self.last_ts = timestamp_ns
        return timestamp_ns


class HostLatencySpikeInjector(BaseFaultInjector):
    """Fault 10: Transport latency spikes (host_time_ns vs device_time_ns)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        spike_delay_ms: float = 80.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.HOST_LATENCY_SPIKE,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            spike_delay_ms=spike_delay_ms,
            **kwargs,
        )
        self.spike_ns = int(round(spike_delay_ms * 1_000_000 * severity))

    def modify_host_time(self, host_time_ns: int, device_time_ns: int) -> int:
        if not self.is_active(device_time_ns):
            return host_time_ns
        return host_time_ns + self.spike_ns


class IMUDropoutInjector(BaseFaultInjector):
    """Fault 11: IMU stream dropout / silence gap."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.IMU_DROPOUT,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            **kwargs,
        )

    def should_drop(self, timestamp_ns: int) -> bool:
        return self.is_active(timestamp_ns)


class StreamDisconnectInjector(BaseFaultInjector):
    """Fault 12: Stream disconnects / terminates mid-session."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.STREAM_DISCONNECT,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            **kwargs,
        )

    def should_drop(self, timestamp_ns: int) -> bool:
        return timestamp_ns >= self.t_start_ns
