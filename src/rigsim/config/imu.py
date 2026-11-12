"""IMU specs, noise models, and saturation limits."""

from pydantic import BaseModel, Field

from rigsim.config.camera import ExtrinsicsSpec


class IMUSaturation(BaseModel):
    """Saturation thresholds for gyroscope and accelerometer."""

    gyro_max_rad_s: float = Field(default=34.906, gt=0.0, description="~2000 deg/s in rad/s")
    accel_max_m_s2: float = Field(default=156.96, gt=0.0, description="~16 g in m/s^2")


class IMUSpec(BaseModel):
    """IMU specification following Kalibr / EuRoC ADIS16448 parameters."""

    stream_id: str = "imu0"
    nominal_rate_hz: float = Field(200.0, gt=0.0)
    clock_id: str
    frame_id: str = "imu_link"

    # Gyroscope noise parameters (EuRoC default: ADIS16448)
    gyro_noise_density: float = Field(1.6968e-04, gt=0.0, description="rad / (s * sqrt(Hz))")
    gyro_random_walk: float = Field(1.9393e-05, gt=0.0, description="rad / (s^2 * sqrt(Hz))")

    # Accelerometer noise parameters (EuRoC default: ADIS16448)
    accel_noise_density: float = Field(2.0e-03, gt=0.0, description="m / (s^2 * sqrt(Hz))")
    accel_random_walk: float = Field(3.0e-03, gt=0.0, description="m / (s^3 * sqrt(Hz))")

    saturation: IMUSaturation = Field(default_factory=lambda: IMUSaturation())
    extrinsics_to_rig: ExtrinsicsSpec
