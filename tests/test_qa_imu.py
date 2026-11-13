"""Unit tests for physical IMU QA checks: M1, M2, M3, and M4."""

import pytest

from rigsim.config.qa import IMUThresholds
from rigsim.config.types import QAVerdict
from rigsim.qa.imu import (
    check_m1_imu_gravity,
    check_m2_imu_static_bias,
    check_m3_imu_bias_jump,
    check_m4_imu_gravity_direction,
)


@pytest.fixture
def default_thresholds() -> IMUThresholds:
    return IMUThresholds(
        max_gravity_error_m_s2=0.20,
        max_gyro_static_bias_rad_s=0.01,
        max_accel_static_bias_m_s2=0.10,
        max_saturation_samples=0,
    )


def test_m1_imu_gravity(default_thresholds: IMUThresholds) -> None:
    # 200 static samples at exactly 9.81 m/s^2 on z
    accel = [[0.0, 0.0, 9.80665] for _ in range(200)]
    gyro = [[0.001, -0.001, 0.0] for _ in range(200)]
    ts = [int(1e9 + i * 5_000_000) for i in range(200)]

    res_clean = check_m1_imu_gravity("imu0", accel, gyro, ts, default_thresholds)
    assert res_clean.verdict == QAVerdict.PASS
    assert res_clean.value < 0.05

    # Wrong gravity norm: 10.4 m/s^2
    accel_bad = [[0.0, 0.0, 10.4] for _ in range(200)]
    res_bad = check_m1_imu_gravity("imu0", accel_bad, gyro, ts, default_thresholds)
    assert res_bad.verdict == QAVerdict.FAIL
    assert res_bad.value > 0.20


def test_m2_imu_static_bias(default_thresholds: IMUThresholds) -> None:
    accel = [[0.0, 0.0, 9.80665] for _ in range(200)]
    ts = [int(1e9 + i * 5_000_000) for i in range(200)]

    # Clean bias: 0.002 rad/s (< 0.01 threshold)
    gyro_clean = [[0.001, 0.001, 0.001] for _ in range(200)]
    res_clean = check_m2_imu_static_bias("imu0", accel, gyro_clean, ts, default_thresholds)
    assert res_clean.verdict == QAVerdict.PASS
    assert res_clean.value < 0.01

    # Large gyro bias: 0.05 rad/s
    gyro_bad = [[0.03, 0.03, 0.03] for _ in range(200)]
    res_bad = check_m2_imu_static_bias("imu0", accel, gyro_bad, ts, default_thresholds)
    assert res_bad.verdict == QAVerdict.FAIL
    assert res_bad.value > 0.01


def test_m3_imu_bias_jump() -> None:
    accel = [[0.0, 0.0, 9.80665] for _ in range(200)]
    ts = [int(1e9 + i * 5_000_000) for i in range(200)]

    # Stable bias
    gyro_clean = [[0.001, 0.0, 0.0] for _ in range(200)]
    res_clean = check_m3_imu_bias_jump("imu0", accel, gyro_clean, ts)
    assert res_clean.verdict == QAVerdict.PASS

    # Bias jump at sample 100: shift by +0.06 rad/s
    gyro_jump = [[0.001, 0.0, 0.0] for _ in range(100)] + [[0.061, 0.0, 0.0] for _ in range(100)]
    res_jump = check_m3_imu_bias_jump("imu0", accel, gyro_jump, ts)
    assert res_jump.verdict == QAVerdict.FAIL
    assert res_jump.value > 0.02
    assert len(res_jump.flagged_windows) >= 1


def test_m4_imu_gravity_direction() -> None:
    # Rest vector: [0, 0, +9.81] (reaction to gravity pointing up +Z)
    accel_upright = [[0.0, 0.0, 9.80665] for _ in range(100)]
    gyro = [[0.0, 0.0, 0.0] for _ in range(100)]
    ts = [int(1e9 + i * 5_000_000) for i in range(100)]

    res_good = check_m4_imu_gravity_direction("imu0", accel_upright, gyro, ts)
    assert res_good.verdict == QAVerdict.PASS
    assert res_good.value < 1.0

    # Inverted / flipped Z axis: [0, 0, -9.81]
    accel_flipped = [[0.0, 0.0, -9.80665] for _ in range(100)]
    res_flipped = check_m4_imu_gravity_direction("imu0", accel_flipped, gyro, ts)
    assert res_flipped.verdict == QAVerdict.FAIL
    assert res_flipped.value > 150.0  # ~180 degrees error
