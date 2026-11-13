"""Unit tests for modern OpenCV ChArUco calibration (C1)."""

import cv2
import numpy as np

from rigsim.calib.charuco import CharucoBoardSpec, CharucoCalibrator


def test_charuco_detector_synthetic_board() -> None:
    calib = CharucoCalibrator(CharucoBoardSpec(squares_x=7, squares_y=5))
    img = calib.board.generateImage((640, 480))

    corners, ids, obj_pts, img_pts = calib.detect_board(img)
    assert corners is not None
    assert ids is not None
    assert obj_pts is not None
    assert img_pts is not None
    assert len(corners) == 24
    assert len(obj_pts) == 24
    assert len(img_pts) == 24
    assert obj_pts.shape[1] == 3
    assert img_pts.shape[1] == 2


def test_charuco_mono_calibration() -> None:
    calib = CharucoCalibrator(CharucoBoardSpec(squares_x=7, squares_y=5))
    base_img = calib.board.generateImage((640, 480))

    # Generate 5 slightly perturbed synthetic views
    views = [base_img]
    for angle in [-15, -8, 8, 15]:
        center = (320, 240)
        M = cv2.getRotationMatrix2D(center, angle, 0.9)
        warped = cv2.warpAffine(base_img, M, (640, 480), borderValue=(200, 200, 200))
        views.append(warped)

    res = calib.calibrate_mono(views, (640, 480))
    assert res.rms_reprojection_px < 0.50
    assert res.K.shape == (3, 3)
    assert res.K[0, 0] > 0
    assert res.num_views >= 3
    assert len(res.per_view_errors) == res.num_views


def test_charuco_stereo_calibration() -> None:
    calib = CharucoCalibrator(CharucoBoardSpec(squares_x=7, squares_y=5))
    base_img = calib.board.generateImage((640, 480))

    left_views = []
    right_views = []

    # Simulate horizontal stereo shift: right camera shifted by translation
    for shift_x in [0, 10, -10]:
        M_l = np.float32([[1, 0, shift_x], [0, 1, 0]])
        im_l = cv2.warpAffine(base_img, M_l, (640, 480), borderValue=(200, 200, 200))

        # Right camera has horizontal baseline disparity shift
        M_r = np.float32([[1, 0, shift_x - 30], [0, 1, 0]])
        im_r = cv2.warpAffine(base_img, M_r, (640, 480), borderValue=(200, 200, 200))

        left_views.append(im_l)
        right_views.append(im_r)

    mono_l = calib.calibrate_mono(left_views, (640, 480))
    mono_r = calib.calibrate_mono(right_views, (640, 480))

    stereo_res = calib.calibrate_stereo(
        left_views,
        right_views,
        (640, 480),
        mono_l.K,
        mono_l.distortion,
        mono_r.K,
        mono_r.distortion,
    )

    assert stereo_res.rms_stereo_px < 0.70
    assert stereo_res.baseline_m > 0.0
    assert stereo_res.R.shape == (3, 3)
    assert stereo_res.T.shape == (3, 1)
    assert stereo_res.num_stereo_pairs == len(left_views)

