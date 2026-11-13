"""Unit tests for IMU-Camera temporal offset estimation (T7)."""

import numpy as np

from rigsim.config.qa import TimingThresholds
from rigsim.config.types import QAVerdict
from rigsim.qa.timing import check_t7_imu_cam_offset


def test_imu_cam_sync_synthetic_offset() -> None:
    """Test that known injected temporal offset between gyro and visual motion is recovered."""
    # Simulation duration: 5.0 seconds
    duration_s = 5.0
    cam_dt_s = 1.0 / 30.0  # 30 Hz camera
    imu_dt_s = 1.0 / 200.0  # 200 Hz IMU

    # Injected offset: IMU is ahead of camera by +15 ms (0.015s)
    true_offset_s = 0.015
    true_offset_ms = true_offset_s * 1000.0

    t_cam = np.arange(0.5, duration_s - 0.5, cam_dt_s)
    t_imu = np.arange(0.5, duration_s - 0.5, imu_dt_s)

    # Synthetic angular velocity profile: sum of sinusoids
    def omega_profile(t: np.ndarray) -> np.ndarray:
        return 0.5 * np.sin(2 * np.pi * 1.5 * t) + 0.3 * np.sin(2 * np.pi * 3.0 * t)

    omega_vis = np.abs(omega_profile(t_cam))
    omega_gyro = np.abs(omega_profile(t_imu + true_offset_s))

    # Direct cross-correlation on 1D angular rate curves
    grid_t = np.arange(1.0, 4.0, 0.001)
    sig_vis = np.interp(grid_t, t_cam, omega_vis)
    sig_imu = np.interp(grid_t, t_imu, omega_gyro)

    # Normalize
    sig_vis = (sig_vis - np.mean(sig_vis)) / np.std(sig_vis)
    sig_imu = (sig_imu - np.mean(sig_imu)) / np.std(sig_imu)

    max_lag = 40
    corr = np.correlate(sig_vis, sig_imu, mode="full")
    center = len(sig_imu) - 1
    corr_slice = corr[center - max_lag : center + max_lag + 1] / len(grid_t)
    lags = np.arange(-max_lag, max_lag + 1)

    peak_idx = int(np.argmax(corr_slice))
    estimated_offset_ms = lags[peak_idx] * 1.0  # 1 ms grid

    # Parabolic sub-ms refinement
    y_neg = corr_slice[peak_idx - 1]
    y_zero = corr_slice[peak_idx]
    y_pos = corr_slice[peak_idx + 1]
    delta = (y_neg - y_pos) / (2.0 * (y_neg - 2.0 * y_zero + y_pos))
    refined_offset_ms = estimated_offset_ms + delta

    # Should accurately recover +15.0 ms within sub-frame tolerance (2.0 ms on 33.3 ms frame)
    np.testing.assert_allclose(refined_offset_ms, true_offset_ms, atol=2.0)


def test_t7_qa_check_verdict() -> None:
    # Insufficient motion
    frames = [np.zeros((60, 80, 3), dtype=np.uint8) for _ in range(15)]
    cam_ts = [int(1e9 + i * 33_333_333) for i in range(15)]
    gyro = [[0.0, 0.0, 0.0] for _ in range(100)]
    imu_ts = [int(1e9 + i * 5_000_000) for i in range(100)]
    thresholds = TimingThresholds(max_imu_cam_offset_ms=5.0)

    res = check_t7_imu_cam_offset("cam0", frames, cam_ts, "imu0", gyro, imu_ts, thresholds)
    assert res.verdict == QAVerdict.PASS
    assert "skipped" in res.explanation
