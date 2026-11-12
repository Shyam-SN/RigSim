"""IMU content fault injectors with ground-truth label generation."""

from typing import Any

import numpy as np

from rigsim.config.types import FaultType
from rigsim.faults.base import BaseFaultInjector
from rigsim.sim.imu import IMUSample


class IMUSaturationInjector(BaseFaultInjector):
    """Fault 21: Sensor clipping / saturation at physical range limits."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        axis: str = "gyro_z",  # gyro_z, accel_z, etc.
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.IMU_SATURATION,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            axis=axis,
            **kwargs,
        )
        self.axis = axis

    def modify_sample(self, sample: IMUSample) -> IMUSample:
        if not self.is_active(sample.device_time_ns):
            return sample

        gyro = sample.angular_velocity.copy()
        accel = sample.linear_acceleration.copy()

        if self.axis.startswith("gyro"):
            gyro[2] = 34.906  # Clip at maximum 2000 deg/s
        else:
            accel[2] = 156.96  # Clip at maximum 16g

        return IMUSample(
            device_time_ns=sample.device_time_ns,
            host_time_ns=sample.host_time_ns,
            linear_acceleration=accel,
            angular_velocity=gyro,
            orientation_xyzw=sample.orientation_xyzw,
            temperature_c=sample.temperature_c,
            is_saturated=True,
        )


class IMUBiasJumpInjector(BaseFaultInjector):
    """Fault 22: Sudden bias jump / DC offset shift in gyroscope or accelerometer."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        accel_jump_m_s2: float = 0.5,
        gyro_jump_rad_s: float = 0.05,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.IMU_BIAS_JUMP,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            accel_jump_m_s2=accel_jump_m_s2,
            gyro_jump_rad_s=gyro_jump_rad_s,
            **kwargs,
        )
        self.accel_delta = np.array([0.2, -0.3, accel_jump_m_s2]) * severity
        self.gyro_delta = np.array([0.02, 0.01, gyro_jump_rad_s]) * severity

    def modify_sample(self, sample: IMUSample) -> IMUSample:
        if sample.device_time_ns < self.t_start_ns:
            return sample

        return IMUSample(
            device_time_ns=sample.device_time_ns,
            host_time_ns=sample.host_time_ns,
            linear_acceleration=sample.linear_acceleration + self.accel_delta,
            angular_velocity=sample.angular_velocity + self.gyro_delta,
            orientation_xyzw=sample.orientation_xyzw,
            temperature_c=sample.temperature_c,
            is_saturated=sample.is_saturated,
        )


class IMUHighNoiseInjector(BaseFaultInjector):
    """Fault 23: Increased stochastic noise (high temperature / electrical interference)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        noise_multiplier: float = 8.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.IMU_HIGH_NOISE,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            noise_multiplier=noise_multiplier,
            **kwargs,
        )
        self.multiplier = noise_multiplier * severity

    def modify_sample(self, sample: IMUSample) -> IMUSample:
        if not self.is_active(sample.device_time_ns):
            return sample

        accel_noise = self.rng.normal(0.0, 0.1 * self.multiplier, size=3)
        gyro_noise = self.rng.normal(0.0, 0.02 * self.multiplier, size=3)

        return IMUSample(
            device_time_ns=sample.device_time_ns,
            host_time_ns=sample.host_time_ns,
            linear_acceleration=sample.linear_acceleration + accel_noise,
            angular_velocity=sample.angular_velocity + gyro_noise,
            orientation_xyzw=sample.orientation_xyzw,
            temperature_c=sample.temperature_c,
            is_saturated=sample.is_saturated,
        )


class IMUAxisFlipInjector(BaseFaultInjector):
    """Fault 24: Coordinate axis flip or frame misconfiguration (e.g. z -> -z)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        flip_axis: str = "z",  # Invert z-axis (catches gravity sign error)
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.IMU_AXIS_FLIP,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            flip_axis=flip_axis,
            **kwargs,
        )
        self.flip_axis = flip_axis

    def modify_sample(self, sample: IMUSample) -> IMUSample:
        if sample.device_time_ns < self.t_start_ns:
            return sample

        accel = sample.linear_acceleration.copy()
        gyro = sample.angular_velocity.copy()

        if self.flip_axis == "z":
            accel[2] = -accel[2]
            gyro[2] = -gyro[2]
        elif self.flip_axis == "x":
            accel[0] = -accel[0]
            gyro[0] = -gyro[0]
        elif self.flip_axis == "y":
            accel[1] = -accel[1]
            gyro[1] = -gyro[1]

        return IMUSample(
            device_time_ns=sample.device_time_ns,
            host_time_ns=sample.host_time_ns,
            linear_acceleration=accel,
            angular_velocity=gyro,
            orientation_xyzw=sample.orientation_xyzw,
            temperature_c=sample.temperature_c,
            is_saturated=sample.is_saturated,
        )
