"""IMU sensor simulation with Kalibr / EuRoC stochastic noise model."""

from dataclasses import dataclass

import numpy as np

from rigsim.config.imu import IMUSpec


@dataclass
class IMUSample:
    device_time_ns: int
    host_time_ns: int
    linear_acceleration: np.ndarray  # [ax, ay, az] in m/s^2 (includes gravity)
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    orientation_xyzw: np.ndarray  # Ground truth [x, y, z, w]
    temperature_c: float
    is_saturated: bool = False


class IMUSimulator:
    """Simulates an inertial measurement unit following Kalibr / Allan variance noise physics."""

    def __init__(self, spec: IMUSpec, seed: int = 42):
        self.spec = spec
        self.dt = 1.0 / spec.nominal_rate_hz
        self.rng = np.random.default_rng(seed)

        # Discrete-time white noise standard deviations: sigma_d = sigma / sqrt(dt)
        self.gyro_white_std = spec.gyro_noise_density / np.sqrt(self.dt)
        self.accel_white_std = spec.accel_noise_density / np.sqrt(self.dt)

        # Discrete-time random walk standard deviations: sigma_b_d = sigma_b * sqrt(dt)
        self.gyro_walk_std = spec.gyro_random_walk * np.sqrt(self.dt)
        self.accel_walk_std = spec.accel_random_walk * np.sqrt(self.dt)

        # Dynamic bias states (random walk integration)
        self.gyro_bias = self.rng.normal(0.0, 0.005, size=3)  # Initial turn-on bias
        self.accel_bias = self.rng.normal(0.0, 0.05, size=3)

        # Ambient sensor temperature model (starts at 24C, slowly rises to 35C)
        self.temperature_c = 24.5

    def update_and_sample(
        self,
        true_accel: np.ndarray,
        true_gyro: np.ndarray,
        true_quat_wxyz: np.ndarray,
        device_time_ns: int,
        host_time_ns: int,
        t_sim_s: float,
    ) -> IMUSample:
        """Sample IMU with physical stochastic noise and saturation checks."""
        # Step bias random walks: b_k = b_{k-1} + N(0, sigma_b_d^2)
        self.gyro_bias += self.rng.normal(0.0, self.gyro_walk_std, size=3)
        self.accel_bias += self.rng.normal(0.0, self.accel_walk_std, size=3)

        # Add white measurement noise
        gyro_noise = self.rng.normal(0.0, self.gyro_white_std, size=3)
        accel_noise = self.rng.normal(0.0, self.accel_white_std, size=3)

        noisy_gyro = true_gyro + self.gyro_bias + gyro_noise
        noisy_accel = true_accel + self.accel_bias + accel_noise

        # Check saturation limits
        is_saturated = False
        gyro_max = self.spec.saturation.gyro_max_rad_s
        accel_max = self.spec.saturation.accel_max_m_s2

        if np.any(np.abs(noisy_gyro) > gyro_max):
            noisy_gyro = np.clip(noisy_gyro, -gyro_max, gyro_max)
            is_saturated = True

        if np.any(np.abs(noisy_accel) > accel_max):
            noisy_accel = np.clip(noisy_accel, -accel_max, accel_max)
            is_saturated = True

        # Slowly warming temperature
        self.temperature_c = 24.5 + 8.0 * (1.0 - np.exp(-t_sim_s / 120.0))

        # Convert MuJoCo [w, x, y, z] to ROS/OpenCV [x, y, z, w]
        quat_xyzw = np.array(
            [true_quat_wxyz[1], true_quat_wxyz[2], true_quat_wxyz[3], true_quat_wxyz[0]],
            dtype=np.float64,
        )

        return IMUSample(
            device_time_ns=device_time_ns,
            host_time_ns=host_time_ns,
            linear_acceleration=noisy_accel,
            angular_velocity=noisy_gyro,
            orientation_xyzw=quat_xyzw,
            temperature_c=float(self.temperature_c),
            is_saturated=is_saturated,
        )
