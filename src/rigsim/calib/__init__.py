"""Calibration module for fiducial detection, camera calibration, and geometric QA."""

from rigsim.calib.charuco import (
    CharucoBoardSpec,
    CharucoCalibrator,
    MonoCalibResult,
    StereoCalibResult,
)
from rigsim.calib.checker import (
    check_c1_charuco_reprojection,
    check_c2_camera_bump,
    check_c3_stereo_consistency,
    check_c4_calibration_staleness,
)

__all__ = [
    "CharucoBoardSpec",
    "CharucoCalibrator",
    "MonoCalibResult",
    "StereoCalibResult",
    "check_c1_charuco_reprojection",
    "check_c2_camera_bump",
    "check_c3_stereo_consistency",
    "check_c4_calibration_staleness",
]

