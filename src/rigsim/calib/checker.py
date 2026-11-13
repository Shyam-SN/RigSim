"""Geometric calibration quality assurance checks: C1, C2, C3, and C4."""

from datetime import UTC, datetime

import cv2
import numpy as np

from rigsim.calib.charuco import CharucoCalibrator
from rigsim.config.qa import CalibrationThresholds, QAResult
from rigsim.config.types import QAVerdict


def check_c1_charuco_reprojection(
    stream_id: str,
    rms_px: float,
    thresholds: CalibrationThresholds,
) -> QAResult:
    """C1: ChArUco mono calibration reprojection RMS error."""
    verdict = (
        QAVerdict.FAIL if rms_px > thresholds.max_charuco_reprojection_rms_px else QAVerdict.PASS
    )
    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: ChArUco reprojection error {rms_px:.3f} px exceeds limit "
            f"{thresholds.max_charuco_reprojection_rms_px:.2f} px."
        )
    else:
        explanation = (
            f"ChArUco reprojection RMS {rms_px:.3f} px satisfies calibration limit "
            f"(<{thresholds.max_charuco_reprojection_rms_px:.2f} px)."
        )

    return QAResult(
        check_id="C1_CHARUCO_REPROJECTION",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="rms_reprojection_px",
        value=float(rms_px),
        threshold=thresholds.max_charuco_reprojection_rms_px,
        explanation=explanation,
    )


def check_c2_camera_bump(
    stream_id: str,
    frames_rgb: list[np.ndarray],
    timestamps_ns: list[int],
    K: np.ndarray,
    dist: np.ndarray,
    thresholds: CalibrationThresholds,
    calibrator: CharucoCalibrator | None = None,
) -> QAResult:
    """C2: Camera bump / physical rig extrinsic deformation detection via solvePnP."""
    if len(frames_rgb) < 5:
        return QAResult(
            check_id="C2_CAMERA_BUMP",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="max_bump_rotation_deg",
            value=0.0,
            threshold=thresholds.max_bump_rotation_deg,
            explanation="Insufficient frames for bump analysis.",
        )

    calib = calibrator or CharucoCalibrator()

    valid_indices = []
    rvecs = []
    tvecs = []

    for i, img in enumerate(frames_rgb):
        _, _, obj_pts, img_pts = calib.detect_board(img)
        if obj_pts is not None and img_pts is not None and len(obj_pts) >= 6:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                valid_indices.append(i)
                rvecs.append(rvec.ravel())
                tvecs.append(tvec.ravel())

    if len(valid_indices) < 3:
        return QAResult(
            check_id="C2_CAMERA_BUMP",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="max_bump_rotation_deg",
            value=0.0,
            threshold=thresholds.max_bump_rotation_deg,
            explanation="ChArUco board not visible for solvePnP extrinsic tracking.",
        )

    # Establish baseline pose from initial valid detections (median of first baseline_count frames)
    baseline_count = max(3, min(10, len(valid_indices)))
    r_base = np.median(rvecs[:baseline_count], axis=0)
    R_base, _ = cv2.Rodrigues(r_base)
    t_base = np.median(tvecs[:baseline_count], axis=0)

    max_rot_deg = 0.0
    max_trans_mm = 0.0
    flagged_windows = []

    for k in range(len(valid_indices)):
        idx = valid_indices[k]
        R_curr, _ = cv2.Rodrigues(rvecs[k])
        t_curr = tvecs[k]

        # Relative rotation: R_rel = R_curr * R_base^T
        R_rel = R_curr @ R_base.T
        tr = np.trace(R_rel)
        cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        rot_deg = float(np.arccos(cos_theta) * 180.0 / np.pi)

        # Relative translation in mm
        trans_mm = float(np.linalg.norm(t_curr - t_base) * 1000.0)

        if rot_deg > max_rot_deg:
            max_rot_deg = rot_deg
        if trans_mm > max_trans_mm:
            max_trans_mm = trans_mm

        if (
            rot_deg > thresholds.max_bump_rotation_deg
            or trans_mm > thresholds.max_bump_translation_mm
        ):
            flagged_windows.append(
                {"t_start_ns": timestamps_ns[idx], "t_end_ns": timestamps_ns[idx]}
            )

    verdict = QAVerdict.FAIL if len(flagged_windows) > 0 else QAVerdict.PASS

    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: Camera bump detected on {len(flagged_windows)} frames "
            f"(max angular shift {max_rot_deg:.2f}° > {thresholds.max_bump_rotation_deg:.2f}°, "
            f"max translation {max_trans_mm:.1f} mm > {thresholds.max_bump_translation_mm:.1f} mm)."
        )
    else:
        explanation = (
            f"Rig rigidity verified: max angular shift {max_rot_deg:.2f}°, "
            f"max translation {max_trans_mm:.1f} mm (within limits)."
        )

    return QAResult(
        check_id="C2_CAMERA_BUMP",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="max_bump_rotation_deg",
        value=float(max_rot_deg),
        threshold=thresholds.max_bump_rotation_deg,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={
            "max_rotation_deg": max_rot_deg,
            "max_translation_mm": max_trans_mm,
            "flagged_frames_count": len(flagged_windows),
        },
        explanation=explanation,
    )


def check_c3_stereo_consistency(
    pair_id: str,
    left_images: list[np.ndarray],
    right_images: list[np.ndarray],
    timestamps_ns: list[int],
    F: np.ndarray,
    thresholds: CalibrationThresholds,
    calibrator: CharucoCalibrator | None = None,
) -> QAResult:
    """C3: Stereo epipolar consistency and baseline rigidity check."""
    calib = calibrator or CharucoCalibrator()

    epipolar_errors_px = []
    flagged_windows = []

    for i in range(len(left_images)):
        _, ids_l, _, img_l = calib.detect_board(left_images[i])
        _, ids_r, _, img_r = calib.detect_board(right_images[i])

        if ids_l is not None and ids_r is not None and img_l is not None and img_r is not None:
            common = np.intersect1d(ids_l.flatten(), ids_r.flatten())
            if len(common) >= 4:
                mask_l = np.isin(ids_l.flatten(), common)
                mask_r = np.isin(ids_r.flatten(), common)
                pts_l = img_l[mask_l].reshape(-1, 2)
                pts_r = img_r[mask_r].reshape(-1, 2)

                # Compute epipolar line in right image: l_r = F * x_l
                pts_l_homo = np.hstack([pts_l, np.ones((len(pts_l), 1))])
                pts_r_homo = np.hstack([pts_r, np.ones((len(pts_r), 1))])

                lines_r = (F @ pts_l_homo.T).T  # (N, 3)
                norm_r = np.sqrt(lines_r[:, 0] ** 2 + lines_r[:, 1] ** 2)
                dists = np.abs(np.sum(pts_r_homo * lines_r, axis=1)) / np.maximum(1e-6, norm_r)

                mean_err = float(np.mean(dists))
                epipolar_errors_px.append(mean_err)

                if mean_err > thresholds.max_epipolar_error_px:
                    flagged_windows.append(
                        {"t_start_ns": timestamps_ns[i], "t_end_ns": timestamps_ns[i]}
                    )

    if not epipolar_errors_px:
        return QAResult(
            check_id="C3_STEREO_CONSISTENCY",
            stream_id=pair_id,
            verdict=QAVerdict.PASS,
            metric_name="mean_epipolar_error_px",
            value=0.0,
            threshold=thresholds.max_epipolar_error_px,
            explanation="No common stereo fiducial points visible.",
        )

    max_err = float(np.max(epipolar_errors_px))
    mean_err = float(np.mean(epipolar_errors_px))
    verdict = QAVerdict.FAIL if max_err > thresholds.max_epipolar_error_px else QAVerdict.PASS

    if verdict == QAVerdict.FAIL:
        explanation = (
            f"FAIL: Stereo epipolar constraint violated on {len(flagged_windows)} frames "
            f"(max error {max_err:.2f} px > limit {thresholds.max_epipolar_error_px:.1f} px)."
        )
    else:
        explanation = (
            f"Stereo epipolar consistency verified: mean {mean_err:.2f} px, max {max_err:.2f} px "
            f"(limit {thresholds.max_epipolar_error_px:.1f} px)."
        )

    return QAResult(
        check_id="C3_STEREO_CONSISTENCY",
        stream_id=pair_id,
        verdict=verdict,
        metric_name="max_epipolar_error_px",
        value=max_err,
        threshold=thresholds.max_epipolar_error_px,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"mean_epipolar_error_px": mean_err, "max_epipolar_error_px": max_err},
        explanation=explanation,
    )


def check_c4_calibration_staleness(
    stream_id: str,
    calibration_date: datetime,
    session_date: datetime | None = None,
    max_age_days: float = 7.0,
) -> QAResult:
    """C4: Calibration staleness check according to operational SLA."""
    now = session_date or datetime.now(UTC)
    if calibration_date.tzinfo is None:
        calibration_date = calibration_date.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    age_days = float((now - calibration_date).total_seconds() / 86400.0)
    verdict = QAVerdict.WARN if age_days > max_age_days else QAVerdict.PASS

    if verdict == QAVerdict.WARN:
        explanation = (
            f"WARN: Calibration is stale ({age_days:.1f} days old > SLA limit {max_age_days:.1f} days); "
            f"re-calibration recommended."
        )
    else:
        explanation = (
            f"Calibration current ({age_days:.1f} days old <= limit {max_age_days:.1f} days)."
        )

    return QAResult(
        check_id="C4_CALIBRATION_STALENESS",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="calibration_age_days",
        value=age_days,
        threshold=max_age_days,
        evidence={"age_days": age_days, "calibration_date": calibration_date.isoformat()},
        explanation=explanation,
    )
