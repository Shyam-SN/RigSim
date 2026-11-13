"""Unit tests for geometric calibration checks: C1, C2, C3, and C4."""

from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from rigsim.calib.charuco import CharucoBoardSpec, CharucoCalibrator
from rigsim.calib.checker import (
    check_c1_charuco_reprojection,
    check_c2_camera_bump,
    check_c3_stereo_consistency,
    check_c4_calibration_staleness,
)
from rigsim.config.qa import CalibrationThresholds
from rigsim.config.types import QAVerdict


@pytest.fixture
def default_thresholds() -> CalibrationThresholds:
    return CalibrationThresholds(
        max_charuco_reprojection_rms_px=0.50,
        max_stereo_reprojection_rms_px=0.70,
        max_bump_rotation_deg=0.50,
        max_bump_translation_mm=3.0,
        max_epipolar_error_px=1.0,
    )


def test_c1_charuco_reprojection(default_thresholds: CalibrationThresholds) -> None:
    res_pass = check_c1_charuco_reprojection("cam0", 0.32, default_thresholds)
    assert res_pass.verdict == QAVerdict.PASS
    assert res_pass.value == 0.32

    res_fail = check_c1_charuco_reprojection("cam0", 0.68, default_thresholds)
    assert res_fail.verdict == QAVerdict.FAIL


def test_c2_camera_bump_detection(default_thresholds: CalibrationThresholds) -> None:
    calib = CharucoCalibrator(CharucoBoardSpec(squares_x=7, squares_y=5))
    base_img = calib.board.generateImage((640, 480))

    # Pinhole K
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    dist = np.zeros(5)

    # 10 stationary frames
    frames = [base_img.copy() for _ in range(10)]
    ts = [1_000_000_000 + i * 33_333_333 for i in range(10)]

    res_clean = check_c2_camera_bump("cam0", frames, ts, K, dist, default_thresholds, calib)
    assert res_clean.verdict == QAVerdict.PASS
    assert res_clean.value < 0.1

    # Inject camera bump on frames 6..9 (rotate image by 3.5 degrees)
    bumped_frames = frames.copy()
    M_bump = cv2.getRotationMatrix2D((320, 240), 3.5, 1.0)
    for k in range(6, 10):
        bumped_frames[k] = cv2.warpAffine(base_img, M_bump, (640, 480), borderValue=(200, 200, 200))

    res_bump = check_c2_camera_bump("cam0", bumped_frames, ts, K, dist, default_thresholds, calib)
    assert res_bump.verdict == QAVerdict.FAIL
    assert res_bump.value > 0.50
    assert len(res_bump.flagged_windows) >= 1


def test_c3_stereo_consistency_epipolar(default_thresholds: CalibrationThresholds) -> None:
    calib = CharucoCalibrator(CharucoBoardSpec(squares_x=7, squares_y=5))
    base_img = calib.board.generateImage((640, 480))

    # Clean rectified horizontal stereo: right image shifted purely horizontally by -25px
    M_r = np.float32([[1, 0, -25], [0, 1, 0]])
    im_r_clean = cv2.warpAffine(base_img, M_r, (640, 480), borderValue=(200, 200, 200))

    left_frames = [base_img for _ in range(5)]
    right_frames_clean = [im_r_clean for _ in range(5)]
    ts = [1_000_000_000 + i * 33_333_333 for i in range(5)]

    # Fundamental matrix for pure horizontal translation: F = [0 0 0; 0 0 -1; 0 1 0]
    # x_r^T F x_l = 0 when y_r == y_l
    F = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    res_clean = check_c3_stereo_consistency(
        "pair0", left_frames, right_frames_clean, ts, F, default_thresholds, calib
    )
    assert res_clean.verdict == QAVerdict.PASS
    assert res_clean.value < 0.2

    # Perturb right frames vertically by 4 pixels (epipolar violation)
    M_r_bad = np.float32([[1, 0, -25], [0, 1, 4.0]])
    im_r_bad = cv2.warpAffine(base_img, M_r_bad, (640, 480), borderValue=(200, 200, 200))
    right_frames_bad = [im_r_bad for _ in range(5)]

    res_bad = check_c3_stereo_consistency(
        "pair0", left_frames, right_frames_bad, ts, F, default_thresholds, calib
    )
    assert res_bad.verdict == QAVerdict.FAIL
    assert res_bad.value > 1.0


def test_c4_calibration_staleness() -> None:
    now = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)

    # 3 days old (fresh)
    calib_fresh = now - timedelta(days=3)
    res_fresh = check_c4_calibration_staleness("cam0", calib_fresh, session_date=now, max_age_days=7.0)
    assert res_fresh.verdict == QAVerdict.PASS
    assert res_fresh.value <= 3.1

    # 14 days old (stale)
    calib_stale = now - timedelta(days=14)
    res_stale = check_c4_calibration_staleness("cam0", calib_stale, session_date=now, max_age_days=7.0)
    assert res_stale.verdict == QAVerdict.WARN
    assert res_stale.value >= 13.9

