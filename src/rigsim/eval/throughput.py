"""Performance and execution throughput benchmarking for RigSim automated QA checks."""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from rigsim.config.qa import ImageThresholds, IMUThresholds, TimingThresholds
from rigsim.qa.image import (
    check_i1_blur,
    check_i2_exposure,
    check_i3_occlusion,
    check_i4_frozen_frames,
    check_i5_corruption,
    check_i6_noise_level,
)
from rigsim.qa.imu import (
    check_m1_imu_gravity,
    check_m2_imu_static_bias,
    check_m3_imu_bias_jump,
    check_m4_imu_gravity_direction,
)
from rigsim.qa.timing import (
    check_t1_frame_drops,
    check_t2_effective_fps,
    check_t3_jitter,
    check_t4_timestamp_sanity,
    check_t5_transport_latency,
    check_t8_imu_continuity,
)


@dataclass
class CheckThroughput:
    category: str
    check_id: str
    check_name: str
    samples_processed: int
    duration_s: float
    throughput_rate: float  # frames/sec or samples/sec
    target_realtime_fps: float
    realtime_speedup: float  # throughput_rate / target_realtime_fps


@dataclass
class ThroughputReport:
    benchmarks: list[CheckThroughput]
    total_duration_s: float
    average_image_fps: float
    average_imu_samples_per_s: float


class QAThroughputBenchmarker:
    """Measures processing speed and realtime throughput multipliers across all QA check algorithms."""

    def __init__(self, num_camera_frames: int = 60, num_imu_samples: int = 2000):
        self.num_camera_frames = num_camera_frames
        self.num_imu_samples = num_imu_samples

    def run_benchmark(self) -> ThroughputReport:
        """Execute speed benchmarks on synthetic camera frames and IMU timeseries."""
        rng = np.random.default_rng(42)
        benchmarks: list[CheckThroughput] = []

        # 1. Synthesize realistic 1280x720 video frames
        frames: list[np.ndarray] = []
        for i in range(self.num_camera_frames):
            frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
            cv2.circle(frame, (300 + (i * 5) % 600, 360), 80, (200, 100, 50), -1)
            noise = rng.normal(0, 2, (720, 1280, 3)).astype(np.int16)
            noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(noisy)

        # 2. Synthesize camera timestamps (30 fps)
        fps = 30.0
        dt_ns = int(1e9 / fps)
        t0_ns = 1_700_000_000_000_000_000
        cam_device_times = [t0_ns + i * dt_ns for i in range(self.num_camera_frames)]
        cam_host_times = [t + int(rng.uniform(2e6, 8e6)) for t in cam_device_times]

        # 3. Synthesize IMU samples (200 Hz)
        imu_rate = 200.0
        imu_dt_ns = int(1e9 / imu_rate)
        imu_device_times = [t0_ns + i * imu_dt_ns for i in range(self.num_imu_samples)]
        imu_accel = np.zeros((self.num_imu_samples, 3), dtype=np.float64)
        imu_accel[:, 2] = 9.81 + rng.normal(0, 0.05, self.num_imu_samples)
        imu_gyro = rng.normal(0, 0.005, (self.num_imu_samples, 3))
        imu_is_sat = [False] * self.num_imu_samples

        timing_thresh = TimingThresholds()
        img_thresh = ImageThresholds()
        imu_thresh = IMUThresholds()

        # Benchmarking Timing Checks
        # T1 Frame Drops
        t_start = time.perf_counter()
        _ = check_t1_frame_drops("cam0", cam_device_times, fps, timing_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Timing",
                check_id="T1",
                check_name="Frame Drops",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # T2 Effective FPS
        t_start = time.perf_counter()
        _ = check_t2_effective_fps("cam0", cam_device_times, fps, timing_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Timing",
                check_id="T2",
                check_name="Effective FPS",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # T3 Jitter
        t_start = time.perf_counter()
        _ = check_t3_jitter("cam0", cam_device_times, fps, timing_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Timing",
                check_id="T3",
                check_name="Framerate Jitter",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # T4 Timestamp Sanity
        t_start = time.perf_counter()
        _ = check_t4_timestamp_sanity("cam0", cam_device_times)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Timing",
                check_id="T4",
                check_name="Timestamp Sanity",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # T5 Host Latency
        t_start = time.perf_counter()
        _ = check_t5_transport_latency("cam0", cam_device_times, cam_host_times, timing_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Timing",
                check_id="T5",
                check_name="Transport Latency",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # T8 IMU Continuity
        t_start = time.perf_counter()
        _ = check_t8_imu_continuity("imu0", imu_device_times, imu_is_sat, imu_rate)
        dur = time.perf_counter() - t_start
        rate = self.num_imu_samples / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Timing",
                check_id="T8",
                check_name="IMU Continuity",
                samples_processed=self.num_imu_samples,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=imu_rate,
                realtime_speedup=rate / imu_rate,
            )
        )

        # Benchmarking Image Content Checks (on 1280x720 RGB frames)
        # I1 Blur
        t_start = time.perf_counter()
        _ = check_i1_blur("cam0", frames, cam_device_times, img_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Image",
                check_id="I1",
                check_name="Laplacian Blur",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # I2 Exposure
        t_start = time.perf_counter()
        _ = check_i2_exposure("cam0", frames, cam_device_times, img_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Image",
                check_id="I2",
                check_name="Exposure Histogram",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # I3 Occlusion
        t_start = time.perf_counter()
        _ = check_i3_occlusion("cam0", frames, cam_device_times, img_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Image",
                check_id="I3",
                check_name="Lens Occlusion",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # I4 Frozen Frame
        t_start = time.perf_counter()
        _ = check_i4_frozen_frames("cam0", frames, cam_device_times, img_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Image",
                check_id="I4",
                check_name="Frozen Frames",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # I5 Corruption
        t_start = time.perf_counter()
        _ = check_i5_corruption("cam0", frames, cam_device_times)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Image",
                check_id="I5",
                check_name="Image Corruption",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # I6 Noise Level
        t_start = time.perf_counter()
        _ = check_i6_noise_level("cam0", frames, cam_device_times)
        dur = time.perf_counter() - t_start
        rate = self.num_camera_frames / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="Image",
                check_id="I6",
                check_name="High Noise Level",
                samples_processed=self.num_camera_frames,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=fps,
                realtime_speedup=rate / fps,
            )
        )

        # Benchmarking IMU Physical Integrity Checks
        # M1 Gravity Norm
        t_start = time.perf_counter()
        _ = check_m1_imu_gravity("imu0", imu_accel, imu_gyro, imu_device_times, imu_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_imu_samples / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="IMU",
                check_id="M1",
                check_name="Gravity Norm",
                samples_processed=self.num_imu_samples,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=imu_rate,
                realtime_speedup=rate / imu_rate,
            )
        )

        # M2 Static Bias
        t_start = time.perf_counter()
        _ = check_m2_imu_static_bias("imu0", imu_accel, imu_gyro, imu_device_times, imu_thresh)
        dur = time.perf_counter() - t_start
        rate = self.num_imu_samples / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="IMU",
                check_id="M2",
                check_name="Static Bias",
                samples_processed=self.num_imu_samples,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=imu_rate,
                realtime_speedup=rate / imu_rate,
            )
        )

        # M3 Bias Jump
        t_start = time.perf_counter()
        _ = check_m3_imu_bias_jump("imu0", imu_accel, imu_gyro, imu_device_times)
        dur = time.perf_counter() - t_start
        rate = self.num_imu_samples / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="IMU",
                check_id="M3",
                check_name="Bias Jump",
                samples_processed=self.num_imu_samples,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=imu_rate,
                realtime_speedup=rate / imu_rate,
            )
        )

        # M4 Gravity Direction
        t_start = time.perf_counter()
        _ = check_m4_imu_gravity_direction("imu0", imu_accel, imu_gyro, imu_device_times)
        dur = time.perf_counter() - t_start
        rate = self.num_imu_samples / max(1e-6, dur)
        benchmarks.append(
            CheckThroughput(
                category="IMU",
                check_id="M4",
                check_name="Gravity Direction",
                samples_processed=self.num_imu_samples,
                duration_s=dur,
                throughput_rate=rate,
                target_realtime_fps=imu_rate,
                realtime_speedup=rate / imu_rate,
            )
        )

        total_dur = sum(b.duration_s for b in benchmarks)
        img_bench = [b for b in benchmarks if b.category == "Image"]
        avg_img_fps = float(np.mean([b.throughput_rate for b in img_bench])) if img_bench else 0.0
        imu_bench = [b for b in benchmarks if b.category == "IMU"]
        avg_imu_rate = float(np.mean([b.throughput_rate for b in imu_bench])) if imu_bench else 0.0

        return ThroughputReport(
            benchmarks=benchmarks,
            total_duration_s=total_dur,
            average_image_fps=avg_img_fps,
            average_imu_samples_per_s=avg_imu_rate,
        )
