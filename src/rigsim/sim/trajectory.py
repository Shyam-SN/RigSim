"""Kinematic trajectory generation for mocap bodies."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


@dataclass
class PoseTrajectorySample:
    t_sim_s: float
    position: np.ndarray  # [x, y, z] in meters
    quaternion_wxyz: np.ndarray  # MuJoCo format: [w, x, y, z]
    linear_velocity: np.ndarray  # [vx, vy, vz] in m/s
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s


class TrajectoryGenerator:
    """Generates continuous C2-smooth 6-DoF trajectories with static and excitation segments."""

    def __init__(
        self,
        duration_s: float,
        dt_s: float = 0.001,
        seed: int = 42,
        static_duration_s: float = 2.0,
        excitation_start_s: float = 4.0,
        excitation_duration_s: float = 6.0,
        center_pos: tuple[float, float, float] = (0.0, -0.2, 0.95),
        pos_amplitude: tuple[float, float, float] = (0.15, 0.15, 0.08),
    ):
        self.duration_s = duration_s
        self.dt_s = dt_s
        self.seed = seed
        self.static_duration_s = static_duration_s
        self.excitation_start_s = excitation_start_s
        self.excitation_duration_s = excitation_duration_s
        self.center_pos = np.array(center_pos, dtype=np.float64)
        self.pos_amplitude = np.array(pos_amplitude, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        self._build_trajectory()

    def _build_trajectory(self) -> None:
        """Precompute spline trajectory samples for 1 kHz simulation."""
        self.times = np.arange(0.0, self.duration_s + self.dt_s, self.dt_s)
        n_samples = len(self.times)

        # 1. Waypoint generation for smooth spline
        # Generate anchor points every ~1.5 seconds
        dt_anchor = 1.5
        anchor_times = np.arange(0.0, self.duration_s + dt_anchor, dt_anchor)

        anchor_pos_list = []
        anchor_quats = []

        # Base orientation: looking down along -Z
        q_base = Rotation.from_euler("xyz", [0, 0, 0]).as_quat()  # [x, y, z, w]

        for t in anchor_times:
            if t < self.static_duration_s:
                # Static segment: exactly stationary
                anchor_pos_list.append(self.center_pos.copy())
                anchor_quats.append(q_base.copy())
            else:
                # Smooth random wander around center
                offset = self.rng.uniform(-self.pos_amplitude, self.pos_amplitude)
                pos = self.center_pos + offset
                anchor_pos_list.append(pos)

                # Small orientation deviations (+- 15 degrees)
                euler_rot = self.rng.uniform(-0.25, 0.25, size=3)
                q = Rotation.from_euler("xyz", euler_rot).as_quat()
                anchor_quats.append(q)

        anchor_positions = np.array(anchor_pos_list)
        pos_spline = CubicSpline(anchor_times, anchor_positions, bc_type="clamped")

        # Evaluate positions and linear velocities (analytic derivative)
        self.positions = pos_spline(self.times)
        self.linear_velocities = pos_spline(self.times, 1)

        # 2. Orientation & Angular Velocities with Excitation Segments
        # Excitation window: inject 3-axis rich sinusoidal rotations (omega up to 2 rad/s)
        self.quats_wxyz = np.zeros((n_samples, 4), dtype=np.float64)
        self.angular_velocities = np.zeros((n_samples, 3), dtype=np.float64)

        for i, t in enumerate(self.times):
            if t < self.static_duration_s:
                # Static segment: zero velocity, base orientation
                rot = Rotation.identity()
                self.angular_velocities[i] = np.zeros(3)
            elif (
                self.excitation_start_s
                <= t
                < (self.excitation_start_s + self.excitation_duration_s)
            ):
                # High-excitation segment: multi-axis sinusoidal angular rate
                # Ax = 1.8 rad/s at 1.2 Hz, Ay = 1.5 rad/s at 1.8 Hz, Az = 2.0 rad/s at 2.5 Hz
                t_rel = t - self.excitation_start_s
                wx = 1.8 * np.sin(2.0 * np.pi * 1.2 * t_rel)
                wy = 1.5 * np.sin(2.0 * np.pi * 1.8 * t_rel)
                wz = 2.0 * np.sin(2.0 * np.pi * 2.5 * t_rel)
                self.angular_velocities[i] = [wx, wy, wz]

                # Integrate angle for pose
                ang_x = -(1.8 / (2.0 * np.pi * 1.2)) * (np.cos(2.0 * np.pi * 1.2 * t_rel) - 1.0)
                ang_y = -(1.5 / (2.0 * np.pi * 1.8)) * (np.cos(2.0 * np.pi * 1.8 * t_rel) - 1.0)
                ang_z = -(2.0 / (2.0 * np.pi * 2.5)) * (np.cos(2.0 * np.pi * 2.5 * t_rel) - 1.0)
                rot = Rotation.from_euler("xyz", [ang_x, ang_y, ang_z])
            else:
                # Mild smooth rotation
                t_rel = t
                wx = 0.3 * np.sin(2.0 * np.pi * 0.4 * t_rel)
                wy = 0.25 * np.sin(2.0 * np.pi * 0.5 * t_rel)
                wz = 0.2 * np.cos(2.0 * np.pi * 0.3 * t_rel)
                self.angular_velocities[i] = [wx, wy, wz]

                ang_x = 0.12 * np.sin(2.0 * np.pi * 0.4 * t_rel)
                ang_y = 0.10 * np.cos(2.0 * np.pi * 0.5 * t_rel)
                ang_z = 0.08 * np.sin(2.0 * np.pi * 0.3 * t_rel)
                rot = Rotation.from_euler("xyz", [ang_x, ang_y, ang_z])

            # MuJoCo uses [w, x, y, z], Scipy returns [x, y, z, w]
            q_xyzw = rot.as_quat()
            self.quats_wxyz[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    def get_pose(self, t_sim_s: float) -> PoseTrajectorySample:
        """Sample pose and velocities at any arbitrary simulation time."""
        idx = int(np.clip(round(t_sim_s / self.dt_s), 0, len(self.times) - 1))
        return PoseTrajectorySample(
            t_sim_s=float(self.times[idx]),
            position=self.positions[idx].copy(),
            quaternion_wxyz=self.quats_wxyz[idx].copy(),
            linear_velocity=self.linear_velocities[idx].copy(),
            angular_velocity=self.angular_velocities[idx].copy(),
        )
