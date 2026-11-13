"""Unit tests for timing QA checks (T1, T2, T3, T4, T5, T6, T8)."""

import numpy as np
import pytest

from rigsim.config.qa import QAVerdict, TimingThresholds
from rigsim.qa.timing import (
    check_t1_frame_drops,
    check_t2_effective_fps,
    check_t3_jitter,
    check_t4_timestamp_sanity,
    check_t5_transport_latency,
    check_t6_inter_camera_sync,
    check_t8_imu_continuity,
)


@pytest.fixture
def default_thresholds() -> TimingThresholds:
    return TimingThresholds(
        max_drop_ratio=0.01,
        max_drop_burst_frames=2,
        max_jitter_ms=2.0,
        min_effective_fps_ratio=0.95,
        max_inter_cam_offset_ms=2.0,
        max_inter_cam_drift_ppm=10.0,
        max_transport_latency_spike_ms=25.0,
    )


def test_t1_frame_drops_clean(default_thresholds: TimingThresholds) -> None:
    # 30 Hz, 300 frames, 0 drops
    dt_ns = int(1e9 / 30.0)
    timestamps = [int(1e9 + i * dt_ns) for i in range(300)]
    res = check_t1_frame_drops("cam0", timestamps, 30.0, default_thresholds)
    assert res.verdict == QAVerdict.PASS
    assert res.value == 0.0
    assert len(res.flagged_windows) == 0


def test_t1_frame_drops_burst(default_thresholds: TimingThresholds) -> None:
    # 30 Hz, skip 4 frames at index 50 (gap of 5 * dt_ns)
    dt_ns = int(1e9 / 30.0)
    timestamps = [int(1e9 + i * dt_ns) for i in range(50)] + [
        int(1e9 + i * dt_ns) for i in range(55, 100)
    ]
    res = check_t1_frame_drops("cam0", timestamps, 30.0, default_thresholds)
    assert res.verdict == QAVerdict.FAIL
    assert res.evidence["longest_burst"] >= 1
    assert len(res.flagged_windows) >= 1


def test_t2_effective_fps(default_thresholds: TimingThresholds) -> None:
    # Slow FPS: 20 Hz instead of 30 Hz
    dt_ns = int(1e9 / 20.0)
    timestamps = [int(1e9 + i * dt_ns) for i in range(100)]
    res = check_t2_effective_fps("cam0", timestamps, 30.0, default_thresholds)
    assert res.verdict in [QAVerdict.WARN, QAVerdict.FAIL]
    assert res.value < 28.5


def test_t3_jitter(default_thresholds: TimingThresholds) -> None:
    # High jitter: std = 5 ms (5_000_000 ns)
    rng = np.random.default_rng(42)
    dt_ns = int(1e9 / 30.0)
    base = np.arange(100) * dt_ns + 1_000_000_000
    noise = rng.normal(0, 5_000_000, size=100)
    noisy_ts = sorted((base + noise).astype(int).tolist())

    res = check_t3_jitter("cam0", noisy_ts, 30.0, default_thresholds)
    assert res.verdict == QAVerdict.FAIL
    assert res.value > 2.0


def test_t4_timestamp_sanity_monotonic() -> None:
    # Strictly monotonic
    timestamps = [1_000_000_000 + i * 33_333_333 for i in range(100)]
    res = check_t4_timestamp_sanity("cam0", timestamps)
    assert res.verdict == QAVerdict.PASS
    assert res.value == 0.0


def test_t4_timestamp_sanity_duplicate_and_backward() -> None:
    # Has a duplicate and a backward jump
    timestamps = [1_000_000_000 + i * 33_333_333 for i in range(20)]
    timestamps[10] = timestamps[9]  # duplicate
    timestamps[15] = timestamps[13]  # backward jump

    res = check_t4_timestamp_sanity("cam0", timestamps)
    assert res.verdict == QAVerdict.FAIL
    assert res.evidence["duplicates"] == 1
    assert res.evidence["non_monotonic"] == 1
    assert len(res.flagged_windows) == 2


def test_t5_transport_latency(default_thresholds: TimingThresholds) -> None:
    # Normal transport latency = 5 ms, spike at sample 30 to 50 ms
    dev_times = [1_000_000_000 + i * 33_333_333 for i in range(50)]
    host_times = [t + 5_000_000 for t in dev_times]
    host_times[30] += 45_000_000  # 50 ms total latency spike

    res = check_t5_transport_latency("cam0", dev_times, host_times, default_thresholds)
    assert res.verdict == QAVerdict.WARN
    assert res.evidence["spikes_count"] == 1
    assert len(res.flagged_windows) == 1


def test_t6_inter_cam_sync(default_thresholds: TimingThresholds) -> None:
    # Pair with low offset (0.2 ms)
    ts_ref = [1_000_000_000 + i * 33_333_333 for i in range(100)]
    ts_sec = [t + 200_000 for t in ts_ref]

    res_good = check_t6_inter_camera_sync(
        "cam_left", "cam_right", ts_ref, ts_sec, default_thresholds
    )
    assert res_good.verdict == QAVerdict.PASS

    # Pair with large offset (5.0 ms)
    ts_bad = [t + 5_000_000 for t in ts_ref]
    res_bad = check_t6_inter_camera_sync(
        "cam_left", "cam_right", ts_ref, ts_bad, default_thresholds
    )
    assert res_bad.verdict == QAVerdict.FAIL


def test_t8_imu_continuity() -> None:
    # 200 Hz IMU (5 ms intervals)
    dt_ns = 5_000_000
    timestamps = [1_000_000_000 + i * dt_ns for i in range(200)]
    saturated = [False] * 200

    res_clean = check_t8_imu_continuity("imu0", timestamps, saturated, 200.0)
    assert res_clean.verdict == QAVerdict.PASS

    # Gap of 25 ms
    timestamps_gap = timestamps[:50] + [t + 20_000_000 for t in timestamps[50:]]
    res_gap = check_t8_imu_continuity("imu0", timestamps_gap, saturated, 200.0)
    assert res_gap.verdict == QAVerdict.FAIL
    assert res_gap.evidence["dropout_gaps"] >= 1

    # Saturation
    saturated_dirty = saturated.copy()
    saturated_dirty[30] = True
    res_sat = check_t8_imu_continuity("imu0", timestamps, saturated_dirty, 200.0)
    assert res_sat.verdict == QAVerdict.FAIL
    assert res_sat.evidence["saturation_count"] == 1
