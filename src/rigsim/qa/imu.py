"""IMU physical and calibration QA checks: M1, M2, M3, and M4."""

import numpy as np

from rigsim.config.qa import IMUThresholds, QAResult
from rigsim.config.types import QAVerdict

STANDARD_GRAVITY = 9.80665


def detect_stationary_mask(
    accel_readings: np.ndarray,  # (N, 3) in m/s^2
    gyro_readings: np.ndarray,  # (N, 3) in rad/s
    gyro_thresh_rad_s: float = 0.08,
    accel_std_thresh: float = 0.15,
    window_samples: int = 50,  # ~250 ms at 200 Hz
) -> np.ndarray:
    """Detect static / non-accelerating intervals."""
    gyro_norm = np.linalg.norm(gyro_readings, axis=1)
    is_quiet_gyro = gyro_norm < gyro_thresh_rad_s

    # Rolling std of accel norm
    accel_norm = np.linalg.norm(accel_readings, axis=1)
    if len(accel_norm) < window_samples:
        return np.asarray(is_quiet_gyro, dtype=bool)

    # Fast rolling standard deviation via convolution
    kernel = np.ones(window_samples) / window_samples
    rolling_mean = np.convolve(accel_norm, kernel, mode="same")
    rolling_sq_mean = np.convolve(accel_norm**2, kernel, mode="same")
    rolling_std = np.sqrt(np.maximum(0.0, rolling_sq_mean - rolling_mean**2))

    is_quiet_accel = rolling_std < accel_std_thresh
    return np.asarray(is_quiet_gyro & is_quiet_accel, dtype=bool)


def check_m1_imu_gravity(
    stream_id: str,
    accel_readings: list[list[float]] | np.ndarray,
    gyro_readings: list[list[float]] | np.ndarray,
    timestamps_ns: list[int],
    thresholds: IMUThresholds,
) -> QAResult:
    """M1: Static gravity norm check (|a| == 9.81 m/s^2 during stationary periods)."""
    accel = np.array(accel_readings, dtype=np.float64)
    gyro = np.array(gyro_readings, dtype=np.float64)

    if len(accel) < 20:
        return QAResult(
            check_id="M1_IMU_GRAVITY",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="gravity_error_m_s2",
            value=0.0,
            threshold=thresholds.max_gravity_error_m_s2,
            explanation="Insufficient samples for static gravity check.",
        )

    static_mask = detect_stationary_mask(accel, gyro)
    static_count = int(np.sum(static_mask))

    if static_count < 20:
        # No extended static interval, evaluate overall minimum motion segment
        gyro_norm = np.linalg.norm(gyro, axis=1)
        quietest = np.argsort(gyro_norm)[: min(50, len(gyro_norm))]
        static_accel = accel[quietest]
    else:
        static_accel = accel[static_mask]

    static_norms = np.linalg.norm(static_accel, axis=1)
    mean_g = float(np.mean(static_norms))
    g_err = float(abs(mean_g - STANDARD_GRAVITY))

    verdict = QAVerdict.FAIL if g_err > thresholds.max_gravity_error_m_s2 else QAVerdict.PASS
    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: Static gravity norm error {g_err:.3f} m/s² "
            f"(measured {mean_g:.3f} m/s² vs expected {STANDARD_GRAVITY:.3f} m/s²)."
        )
    else:
        explanation = (
            f"Static gravity verified: measured {mean_g:.3f} m/s² "
            f"(error {g_err:.3f} m/s² < limit {thresholds.max_gravity_error_m_s2:.2f} m/s²)."
        )

    return QAResult(
        check_id="M1_IMU_GRAVITY",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="gravity_error_m_s2",
        value=g_err,
        threshold=thresholds.max_gravity_error_m_s2,
        evidence={
            "measured_gravity": mean_g,
            "expected_gravity": STANDARD_GRAVITY,
            "static_samples": len(static_accel),
        },
        explanation=explanation,
    )


def check_m2_imu_static_bias(
    stream_id: str,
    accel_readings: list[list[float]] | np.ndarray,
    gyro_readings: list[list[float]] | np.ndarray,
    timestamps_ns: list[int],
    thresholds: IMUThresholds,
) -> QAResult:
    """M2: Zero-rate static bias check for gyro and accelerometer."""
    accel = np.array(accel_readings, dtype=np.float64)
    gyro = np.array(gyro_readings, dtype=np.float64)

    static_mask = detect_stationary_mask(accel, gyro)
    static_count = int(np.sum(static_mask))

    if static_count < 20:
        gyro_norm = np.linalg.norm(gyro, axis=1)
        quietest = np.argsort(gyro_norm)[: min(50, len(gyro_norm))]
        static_gyro = gyro[quietest]
        static_accel = accel[quietest]
    else:
        static_gyro = gyro[static_mask]
        static_accel = accel[static_mask]

    # Gyro bias is mean angular rate during static period
    mean_gyro_bias = np.mean(static_gyro, axis=0)
    gyro_bias_norm = float(np.linalg.norm(mean_gyro_bias))

    # Accel bias is deviation from 9.81
    accel_bias = float(abs(np.mean(np.linalg.norm(static_accel, axis=1)) - STANDARD_GRAVITY))

    has_fail = (
        gyro_bias_norm > thresholds.max_gyro_static_bias_rad_s
        or accel_bias > thresholds.max_accel_static_bias_m_s2
    )
    verdict = QAVerdict.FAIL if has_fail else QAVerdict.PASS

    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: Excessive static bias: gyro {gyro_bias_norm:.4f} rad/s "
            f"(limit {thresholds.max_gyro_static_bias_rad_s:.4f}), "
            f"accel {accel_bias:.3f} m/s² (limit {thresholds.max_accel_static_bias_m_s2:.3f})."
        )
    else:
        explanation = (
            f"Static bias verified: gyro {gyro_bias_norm:.4f} rad/s, "
            f"accel {accel_bias:.3f} m/s² (within limits)."
        )

    return QAResult(
        check_id="M2_IMU_STATIC_BIAS",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="gyro_bias_norm_rad_s",
        value=gyro_bias_norm,
        threshold=thresholds.max_gyro_static_bias_rad_s,
        evidence={
            "gyro_bias_vector": mean_gyro_bias.tolist(),
            "gyro_bias_norm": gyro_bias_norm,
            "accel_bias": accel_bias,
        },
        explanation=explanation,
    )


def check_m3_imu_bias_jump(
    stream_id: str,
    accel_readings: list[list[float]] | np.ndarray,
    gyro_readings: list[list[float]] | np.ndarray,
    timestamps_ns: list[int],
    max_jump_gyro_rad_s: float = 0.02,
    max_jump_accel_m_s2: float = 0.20,
) -> QAResult:
    """M3: Detect sudden step jumps in sensor bias indicative of physical/thermal shock."""
    accel = np.array(accel_readings, dtype=np.float64)
    gyro = np.array(gyro_readings, dtype=np.float64)

    static_mask = detect_stationary_mask(accel, gyro)
    static_indices = np.where(static_mask)[0]

    if len(static_indices) < 60:
        return QAResult(
            check_id="M3_IMU_BIAS_JUMP",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="max_gyro_bias_jump",
            value=0.0,
            threshold=max_jump_gyro_rad_s,
            explanation="Insufficient static windows for bias jump detection.",
        )

    # Split static samples into temporal segments (e.g. segments of 50 samples)
    chunk_size = 50
    chunks = [
        static_indices[i : i + chunk_size]
        for i in range(0, len(static_indices) - chunk_size + 1, chunk_size)
    ]

    gyro_means = [np.mean(gyro[ch], axis=0) for ch in chunks]
    accel_norms = [np.mean(np.linalg.norm(accel[ch], axis=1)) for ch in chunks]

    max_gyro_jump = 0.0
    max_accel_jump = 0.0
    flagged_windows = []

    for k in range(1, len(chunks)):
        g_jump = float(np.linalg.norm(gyro_means[k] - gyro_means[k - 1]))
        a_jump = float(abs(accel_norms[k] - accel_norms[k - 1]))

        if g_jump > max_gyro_jump:
            max_gyro_jump = g_jump
        if a_jump > max_accel_jump:
            max_accel_jump = a_jump

        if g_jump > max_jump_gyro_rad_s or a_jump > max_jump_accel_m_s2:
            t_start = timestamps_ns[chunks[k][0]]
            t_end = timestamps_ns[chunks[k][-1]]
            flagged_windows.append({"t_start_ns": t_start, "t_end_ns": t_end})

    verdict = QAVerdict.FAIL if len(flagged_windows) > 0 else QAVerdict.PASS
    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: Detected sudden IMU bias jump: gyro {max_gyro_jump:.4f} rad/s "
            f"(limit {max_jump_gyro_rad_s:.4f}), accel {max_accel_jump:.3f} m/s² (limit {max_jump_accel_m_s2:.3f})."
        )
    else:
        explanation = (
            f"Stable IMU bias verified: max gyro jump {max_gyro_jump:.4f} rad/s, "
            f"accel jump {max_accel_jump:.3f} m/s²."
        )

    return QAResult(
        check_id="M3_IMU_BIAS_JUMP",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="max_gyro_bias_jump",
        value=max_gyro_jump,
        threshold=max_jump_gyro_rad_s,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"max_gyro_jump": max_gyro_jump, "max_accel_jump": max_accel_jump},
        explanation=explanation,
    )


def check_m4_imu_gravity_direction(
    stream_id: str,
    accel_readings: list[list[float]] | np.ndarray,
    gyro_readings: list[list[float]] | np.ndarray,
    timestamps_ns: list[int],
    expected_upright_vector: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),  # +Z reaction when resting
    max_angle_deg: float = 25.0,
) -> QAResult:
    """M4: Gravity vector alignment check (detects inverted mounts or axis flip faults)."""
    accel = np.array(accel_readings, dtype=np.float64)
    gyro = np.array(gyro_readings, dtype=np.float64)

    static_mask = detect_stationary_mask(accel, gyro)
    static_count = int(np.sum(static_mask))

    if static_count < 20:
        gyro_norm = np.linalg.norm(gyro, axis=1)
        quietest = np.argsort(gyro_norm)[: min(50, len(gyro_norm))]
        static_accel = accel[quietest]
    else:
        static_accel = accel[static_mask]

    mean_accel = np.mean(static_accel, axis=0)
    norm = np.linalg.norm(mean_accel)
    if norm < 1.0:
        return QAResult(
            check_id="M4_IMU_GRAVITY_DIR",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="gravity_angle_error_deg",
            value=0.0,
            threshold=max_angle_deg,
            explanation="Near zero-g or freefall motion; gravity direction skipped.",
        )

    u_meas = mean_accel / norm
    u_exp = np.array(expected_upright_vector, dtype=np.float64)
    u_exp = u_exp / np.linalg.norm(u_exp)

    cos_angle = np.clip(np.dot(u_meas, u_exp), -1.0, 1.0)
    angle_deg = float(np.arccos(cos_angle) * 180.0 / np.pi)

    verdict = QAVerdict.FAIL if angle_deg > max_angle_deg else QAVerdict.PASS
    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: Gravity alignment error {angle_deg:.1f}° > {max_angle_deg:.1f}° "
            f"(measured direction {u_meas.round(2).tolist()} vs expected {u_exp.round(2).tolist()}; "
            f"possible axis flip or inverted mount)."
        )
    else:
        explanation = (
            f"Correct gravity orientation verified: angle error {angle_deg:.1f}° "
            f"(limit {max_angle_deg:.1f}°)."
        )

    return QAResult(
        check_id="M4_IMU_GRAVITY_DIR",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="gravity_angle_error_deg",
        value=angle_deg,
        threshold=max_angle_deg,
        evidence={"measured_vector": u_meas.tolist(), "angle_error_deg": angle_deg},
        explanation=explanation,
    )
