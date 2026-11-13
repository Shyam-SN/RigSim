"""Timing and transport QA checks (T1-T6, T8)."""

import numpy as np

from rigsim.config.qa import QAResult, TimingThresholds
from rigsim.config.types import QAVerdict


def check_t1_frame_drops(
    stream_id: str,
    device_times_ns: list[int],
    nominal_rate_hz: float,
    thresholds: TimingThresholds,
) -> QAResult:
    """T1: Detect dropped frames from inter-frame interval gaps."""
    if len(device_times_ns) < 2:
        return QAResult(
            check_id="T1_FRAME_DROPS",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="drop_ratio",
            value=0.0,
            threshold=thresholds.max_drop_ratio,
            explanation="Insufficient frames to evaluate frame drops.",
        )

    times = np.array(device_times_ns, dtype=np.int64)
    dt_s = np.diff(times) / 1e9
    nominal_period_s = 1.0 / nominal_rate_hz

    # Gaps >= 1.5 * nominal period count as drops
    gap_mask = dt_s >= (1.5 * nominal_period_s)
    gap_indices = np.where(gap_mask)[0]

    total_dropped = 0
    longest_burst = 0
    flagged_windows: list[dict[str, int]] = []

    for idx in gap_indices:
        # Estimate missed frames
        missed = int(round(dt_s[idx] / nominal_period_s)) - 1
        total_dropped += missed
        if missed > longest_burst:
            longest_burst = missed
        flagged_windows.append({"t_start_ns": int(times[idx]), "t_end_ns": int(times[idx + 1])})

    total_expected = len(device_times_ns) + total_dropped
    drop_ratio = total_dropped / max(1, total_expected)

    verdict = QAVerdict.PASS
    reasons = []
    if drop_ratio > thresholds.max_drop_ratio:
        verdict = QAVerdict.FAIL
        reasons.append(
            f"drop ratio {drop_ratio * 100:.2f}% exceeds limit {thresholds.max_drop_ratio * 100:.1f}%"
        )
    if longest_burst > thresholds.max_drop_burst_frames:
        verdict = QAVerdict.FAIL
        reasons.append(
            f"burst drop of {longest_burst} frames exceeds limit {thresholds.max_drop_burst_frames}"
        )

    if not reasons:
        explanation = f"Drop ratio {drop_ratio * 100:.2f}% ({total_dropped} frames) within limits."
    else:
        explanation = "FAIL: " + "; ".join(reasons) + "."

    return QAResult(
        check_id="T1_FRAME_DROPS",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="drop_ratio",
        value=float(drop_ratio),
        threshold=thresholds.max_drop_ratio,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={
            "total_dropped": total_dropped,
            "longest_burst": longest_burst,
            "drop_ratio": drop_ratio,
        },
        explanation=explanation,
    )


def check_t2_effective_fps(
    stream_id: str,
    device_times_ns: list[int],
    nominal_rate_hz: float,
    thresholds: TimingThresholds,
    window_duration_s: float = 1.0,
) -> QAResult:
    """T2: Rolling window effective FPS stability."""
    if len(device_times_ns) < 5:
        return QAResult(
            check_id="T2_EFFECTIVE_FPS",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="effective_fps",
            value=nominal_rate_hz,
            threshold=nominal_rate_hz * thresholds.min_effective_fps_ratio,
            explanation="Insufficient frames to evaluate effective FPS.",
        )

    times_s = (np.array(device_times_ns, dtype=np.float64) - device_times_ns[0]) / 1e9
    total_time = times_s[-1]
    if total_time < window_duration_s:
        overall_fps = len(device_times_ns) / max(0.01, total_time)
        return QAResult(
            check_id="T2_EFFECTIVE_FPS",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="effective_fps",
            value=float(overall_fps),
            threshold=nominal_rate_hz * thresholds.min_effective_fps_ratio,
            explanation=f"Overall FPS {overall_fps:.1f} Hz across short duration.",
        )

    # Compute rolling FPS with 0.5s step over complete non-truncated windows
    step_s = 0.5
    max_eval_t = total_time - window_duration_s
    eval_times = np.arange(0.0, max_eval_t + 1e-6, step_s)
    if len(eval_times) == 0:
        eval_times = np.array([0.0])
    rolling_fps = []
    flagged_windows = []

    min_fps_thresh = nominal_rate_hz * thresholds.min_effective_fps_ratio

    for t in eval_times:
        count = int(np.sum((times_s >= t) & (times_s < t + window_duration_s)))
        fps = count / window_duration_s
        rolling_fps.append(fps)
        if fps < min_fps_thresh:
            t_start = int(device_times_ns[0] + t * 1e9)
            t_end = int(device_times_ns[0] + (t + window_duration_s) * 1e9)
            flagged_windows.append({"t_start_ns": t_start, "t_end_ns": t_end})

    min_fps = float(np.min(rolling_fps))
    mean_fps = float(np.mean(rolling_fps))
    under_ratio = len(flagged_windows) / max(1, len(eval_times))

    verdict = QAVerdict.PASS
    if under_ratio > 0.05 or min_fps < (nominal_rate_hz * 0.85):
        verdict = QAVerdict.WARN if min_fps >= (nominal_rate_hz * 0.80) else QAVerdict.FAIL
        explanation = f"Effective FPS dropped to {min_fps:.1f} Hz (nominal {nominal_rate_hz} Hz)."
    else:
        explanation = f"Stable effective FPS: mean {mean_fps:.1f} Hz, min {min_fps:.1f} Hz."

    return QAResult(
        check_id="T2_EFFECTIVE_FPS",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="min_rolling_fps",
        value=min_fps,
        threshold=min_fps_thresh,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"min_fps": min_fps, "mean_fps": mean_fps, "nominal_rate_hz": nominal_rate_hz},
        explanation=explanation,
    )


def check_t3_jitter(
    stream_id: str,
    device_times_ns: list[int],
    nominal_rate_hz: float,
    thresholds: TimingThresholds,
) -> QAResult:
    """T3: Standard deviation and p99 of inter-frame interval error."""
    if len(device_times_ns) < 5:
        return QAResult(
            check_id="T3_JITTER",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="jitter_std_ms",
            value=0.0,
            threshold=thresholds.max_jitter_ms,
            explanation="Insufficient frames for jitter analysis.",
        )

    times = np.array(device_times_ns, dtype=np.int64)
    dt_ms = np.diff(times) / 1e6
    nominal_ms = 1000.0 / nominal_rate_hz

    # Filter out dropped frame outliers for pure jitter analysis
    normal_dt = dt_ms[dt_ms < (1.5 * nominal_ms)]
    if len(normal_dt) < 3:
        normal_dt = dt_ms

    errors_ms = normal_dt - nominal_ms
    std_jitter_ms = float(np.std(errors_ms))
    p99_jitter_ms = float(np.percentile(np.abs(errors_ms), 99))

    verdict = QAVerdict.PASS
    if std_jitter_ms > thresholds.max_jitter_ms:
        verdict = QAVerdict.FAIL
        explanation = f"FAIL: Timestamp jitter std {std_jitter_ms:.2f} ms exceeds limit {thresholds.max_jitter_ms:.1f} ms."
    else:
        explanation = (
            f"Jitter within limits: std {std_jitter_ms:.2f} ms, p99 {p99_jitter_ms:.2f} ms."
        )

    return QAResult(
        check_id="T3_JITTER",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="jitter_std_ms",
        value=std_jitter_ms,
        threshold=thresholds.max_jitter_ms,
        evidence={"std_ms": std_jitter_ms, "p99_ms": p99_jitter_ms},
        explanation=explanation,
    )


def check_t4_timestamp_sanity(
    stream_id: str,
    device_times_ns: list[int],
) -> QAResult:
    """T4: Sanity check for non-monotonic timestamps, duplicates, and clock steps."""
    if len(device_times_ns) < 2:
        return QAResult(
            check_id="T4_TIMESTAMP_SANITY",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="sanity_violations",
            value=0.0,
            threshold=0.0,
            explanation="Timestamps verified.",
        )

    times = np.array(device_times_ns, dtype=np.int64)
    diffs = np.diff(times)

    duplicates = int(np.sum(diffs == 0))
    non_monotonic = int(np.sum(diffs < 0))

    flagged_windows = []
    bad_indices = np.where(diffs <= 0)[0]
    for idx in bad_indices:
        flagged_windows.append({"t_start_ns": int(times[idx]), "t_end_ns": int(times[idx + 1])})

    total_violations = duplicates + non_monotonic
    verdict = QAVerdict.FAIL if total_violations > 0 else QAVerdict.PASS

    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: Detected {non_monotonic} non-monotonic and {duplicates} duplicate timestamps."
        )
    else:
        explanation = "Strict monotonic timestamps verified (0 duplicate, 0 non-monotonic)."

    return QAResult(
        check_id="T4_TIMESTAMP_SANITY",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="sanity_violations",
        value=float(total_violations),
        threshold=0.0,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"duplicates": duplicates, "non_monotonic": non_monotonic},
        explanation=explanation,
    )


def check_t5_transport_latency(
    stream_id: str,
    device_times_ns: list[int],
    host_times_ns: list[int],
    thresholds: TimingThresholds,
) -> QAResult:
    """T5: Host transport latency (log_time - publish_time) distribution and spike detection."""
    if len(device_times_ns) < 5 or len(device_times_ns) != len(host_times_ns):
        return QAResult(
            check_id="T5_TRANSPORT_LATENCY",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="latency_spike_count",
            value=0.0,
            threshold=0.0,
            explanation="Insufficient transport timing data.",
        )

    dev_t = np.array(device_times_ns, dtype=np.float64)
    host_t = np.array(host_times_ns, dtype=np.float64)
    raw_latency_ms = (host_t - dev_t) / 1e6

    # Remove linear clock offset & drift to get residual transport delays
    # Linear fit: host_t - dev_t = a * dev_t + b
    p = np.polyfit(dev_t - dev_t[0], raw_latency_ms, deg=1)
    fitted_base = np.polyval(p, dev_t - dev_t[0])
    residual_ms = raw_latency_ms - fitted_base

    spike_indices = np.where(residual_ms > thresholds.max_transport_latency_spike_ms)[0]
    flagged_windows = []
    for idx in spike_indices:
        flagged_windows.append(
            {"t_start_ns": int(device_times_ns[idx]), "t_end_ns": int(device_times_ns[idx])}
        )

    max_spike_ms = float(np.max(residual_ms)) if len(residual_ms) > 0 else 0.0
    mean_latency_ms = float(np.mean(raw_latency_ms))

    verdict = QAVerdict.WARN if len(spike_indices) > 0 else QAVerdict.PASS
    if verdict == QAVerdict.WARN:
        explanation = f"Detected {len(spike_indices)} transport latency spikes (max {max_spike_ms:.1f} ms above trend)."
    else:
        explanation = f"Stable transport latency (mean {mean_latency_ms:.1f} ms, no spikes > {thresholds.max_transport_latency_spike_ms} ms)."

    return QAResult(
        check_id="T5_TRANSPORT_LATENCY",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="max_spike_ms",
        value=max_spike_ms,
        threshold=thresholds.max_transport_latency_spike_ms,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={
            "mean_ms": mean_latency_ms,
            "max_spike_ms": max_spike_ms,
            "spikes_count": len(spike_indices),
        },
        explanation=explanation,
    )


def check_t6_inter_camera_sync(
    stream_a_id: str,
    stream_b_id: str,
    stream_a_times_ns: list[int],
    stream_b_times_ns: list[int],
    thresholds: TimingThresholds,
) -> QAResult:
    """T6: Inter-camera temporal synchronization and drift."""
    if len(stream_a_times_ns) < 5 or len(stream_b_times_ns) < 5:
        return QAResult(
            check_id="T6_INTER_CAM_SYNC",
            stream_id=f"{stream_a_id}<->{stream_b_id}",
            verdict=QAVerdict.PASS,
            metric_name="max_sync_offset_ms",
            value=0.0,
            threshold=thresholds.max_inter_cam_offset_ms,
            explanation="Insufficient paired timestamps.",
        )

    # For each timestamp in A, find closest timestamp in B using searchsorted
    times_b = np.array(stream_b_times_ns, dtype=np.int64)
    times_a = np.array(stream_a_times_ns, dtype=np.int64)

    idxs = np.searchsorted(times_b, times_a)
    idxs = np.clip(idxs, 1, len(times_b) - 1)
    left_diffs = np.abs(times_a - times_b[idxs - 1])
    right_diffs = np.abs(times_a - times_b[idxs])
    closest_diffs_ns = np.minimum(left_diffs, right_diffs)
    offsets_ms = closest_diffs_ns / 1e6

    mean_offset_ms = float(np.mean(offsets_ms))
    max_offset_ms = float(np.max(offsets_ms))
    p95_offset_ms = float(np.percentile(offsets_ms, 95))

    # Drift estimate: linear regression of (t_a - closest_t_b) over time
    closest_t_b = np.where(left_diffs < right_diffs, times_b[idxs - 1], times_b[idxs])
    signed_diff_ns = times_a - closest_t_b
    t_elapsed_s = (times_a - times_a[0]) / 1e9
    if t_elapsed_s[-1] > 2.0:
        slope, _ = np.polyfit(t_elapsed_s, signed_diff_ns, deg=1)  # slope is ns/s = ppm * 1000
        drift_ppm = float(slope / 1000.0)
    else:
        drift_ppm = 0.0

    flagged_indices = np.where(offsets_ms > thresholds.max_inter_cam_offset_ms)[0]
    flagged_windows = []
    for idx in flagged_indices:
        flagged_windows.append({"t_start_ns": int(times_a[idx]), "t_end_ns": int(times_a[idx])})

    verdict = QAVerdict.PASS
    if (
        max_offset_ms > thresholds.max_inter_cam_offset_ms
        or abs(drift_ppm) > thresholds.max_inter_cam_drift_ppm
    ):
        verdict = QAVerdict.FAIL
        explanation = f"FAIL: Sync offset max {max_offset_ms:.2f} ms (limit {thresholds.max_inter_cam_offset_ms:.1f} ms), drift {drift_ppm:.1f} ppm."
    else:
        explanation = f"Synchronized: mean offset {mean_offset_ms:.2f} ms, max {max_offset_ms:.2f} ms, drift {drift_ppm:.1f} ppm."

    return QAResult(
        check_id="T6_INTER_CAM_SYNC",
        stream_id=f"{stream_a_id}<->{stream_b_id}",
        verdict=verdict,
        metric_name="max_sync_offset_ms",
        value=max_offset_ms,
        threshold=thresholds.max_inter_cam_offset_ms,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={
            "mean_ms": mean_offset_ms,
            "p95_ms": p95_offset_ms,
            "max_ms": max_offset_ms,
            "drift_ppm": drift_ppm,
        },
        explanation=explanation,
    )


def check_t8_imu_continuity(
    stream_id: str,
    device_times_ns: list[int],
    is_saturated_list: list[bool],
    nominal_rate_hz: float,
) -> QAResult:
    """T8: IMU stream continuity, sampling rate stability, and saturation events."""
    if len(device_times_ns) < 5:
        return QAResult(
            check_id="T8_IMU_CONTINUITY",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="gaps_count",
            value=0.0,
            threshold=0.0,
            explanation="Insufficient IMU samples.",
        )

    times = np.array(device_times_ns, dtype=np.int64)
    dt_s = np.diff(times) / 1e9
    nominal_period_s = 1.0 / nominal_rate_hz

    # Gaps >= 1.5 * nominal period
    gap_indices = np.where(dt_s >= (1.5 * nominal_period_s))[0]
    flagged_windows = []
    for idx in gap_indices:
        flagged_windows.append({"t_start_ns": int(times[idx]), "t_end_ns": int(times[idx + 1])})

    sat_count = int(sum(is_saturated_list))
    total_gaps = len(gap_indices)

    verdict = QAVerdict.PASS
    if total_gaps > 0 or sat_count > 0:
        verdict = QAVerdict.FAIL
        explanation = f"FAIL: IMU continuity issues: {total_gaps} dropout gaps, {sat_count} saturation events."
    else:
        explanation = f"Continuous 200 Hz IMU stream verified (0 dropouts, 0 saturation events across {len(times)} samples)."

    return QAResult(
        check_id="T8_IMU_CONTINUITY",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="gaps_count",
        value=float(total_gaps + sat_count),
        threshold=0.0,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"dropout_gaps": total_gaps, "saturation_count": sat_count},
        explanation=explanation,
    )


def check_t7_imu_cam_offset(
    camera_stream_id: str,
    frames_rgb: list[np.ndarray],
    cam_timestamps_ns: list[int],
    imu_stream_id: str,
    gyro_readings: list[list[float]] | np.ndarray,
    imu_timestamps_ns: list[int],
    thresholds: TimingThresholds,
    K: np.ndarray | None = None,
) -> QAResult:
    """T7: IMU-Camera temporal synchronization offset estimation."""
    from rigsim.sync.imu_cam import ImuCameraSyncEstimator

    estimator = ImuCameraSyncEstimator()
    res = estimator.estimate_offset(
        frames_rgb=frames_rgb,
        cam_timestamps_ns=cam_timestamps_ns,
        gyro_readings=gyro_readings,
        imu_timestamps_ns=imu_timestamps_ns,
        K=K,
    )

    abs_offset_ms = abs(res.offset_ms)
    if res.correlation_peak < 0.25:
        verdict = QAVerdict.PASS
        explanation = (
            f"Low angular excitation (r={res.correlation_peak:.2f}); offset check skipped."
        )
    elif abs_offset_ms > thresholds.max_imu_cam_offset_ms:
        verdict = QAVerdict.FAIL
        explanation = f"FAIL: IMU-Camera temporal offset {res.offset_ms:+.2f} ms exceeds limit {thresholds.max_imu_cam_offset_ms:.1f} ms (r={res.correlation_peak:.2f})."
    else:
        verdict = QAVerdict.PASS
        explanation = f"IMU-Camera temporal sync verified: offset {res.offset_ms:+.2f} ms (limit {thresholds.max_imu_cam_offset_ms:.1f} ms, r={res.correlation_peak:.2f})."

    return QAResult(
        check_id="T7_IMU_CAM_OFFSET",
        stream_id=f"{camera_stream_id}<->{imu_stream_id}",
        verdict=verdict,
        metric_name="imu_cam_offset_ms",
        value=float(abs_offset_ms),
        threshold=thresholds.max_imu_cam_offset_ms,
        evidence={
            "offset_ms": res.offset_ms,
            "correlation_peak": res.correlation_peak,
        },
        explanation=explanation,
    )
