"""ChArUco fiducial detection, intrinsic camera calibration, and stereo calibration."""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MonoCalibResult:
    rms_reprojection_px: float
    K: np.ndarray
    distortion: np.ndarray
    num_views: int
    num_corners_total: int
    per_view_errors: list[float] = field(default_factory=list)


@dataclass
class StereoCalibResult:
    rms_stereo_px: float
    K_left: np.ndarray
    dist_left: np.ndarray
    K_right: np.ndarray
    dist_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    baseline_m: float
    E: np.ndarray
    F: np.ndarray
    num_stereo_pairs: int


@dataclass
class CharucoBoardSpec:
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.04
    marker_length_m: float = 0.02
    dict_id: int = cv2.aruco.DICT_5X5_250


class CharucoCalibrator:
    """Modern OpenCV (4.7+ / 5.0+) ChArUco fiducial detector and calibration solver."""

    def __init__(self, spec: CharucoBoardSpec | None = None):
        self.spec = spec or CharucoBoardSpec()
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.spec.dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (self.spec.squares_x, self.spec.squares_y),
            self.spec.square_length_m,
            self.spec.marker_length_m,
            self.dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def detect_board(
        self, image_rgb_or_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Detect ChArUco corners and return (corners, ids, obj_points, img_points)."""
        gray = (
            cv2.cvtColor(image_rgb_or_bgr, cv2.COLOR_BGR2GRAY)
            if image_rgb_or_bgr.ndim == 3
            else image_rgb_or_bgr
        )

        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        if corners is None or ids is None or len(corners) < 4:
            return None, None, None, None

        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        if obj_points is None or img_points is None:
            return None, None, None, None

        obj_f32 = np.ascontiguousarray(obj_points, dtype=np.float32)
        img_f32 = np.ascontiguousarray(img_points, dtype=np.float32)
        return corners, ids, obj_f32, img_f32

    def calibrate_mono(
        self,
        images: list[np.ndarray],
        image_size: tuple[int, int],  # (width, height)
    ) -> MonoCalibResult:
        """Calibrate single camera intrinsics and distortion from a list of frames."""
        all_obj_points = []
        all_img_points = []
        total_corners = 0

        for img in images:
            _, _, obj_pts, img_pts = self.detect_board(img)
            if obj_pts is not None and img_pts is not None and len(obj_pts) >= 6:
                all_obj_points.append(obj_pts)
                all_img_points.append(img_pts)
                total_corners += len(obj_pts)

        if len(all_obj_points) < 1:
            raise ValueError("Insufficient ChArUco board detections across images (need at least 1 view with >= 6 corners).")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points, all_img_points, image_size, None, None
        )

        # Compute per-view reprojection errors
        per_view_errors = []
        for i in range(len(all_obj_points)):
            proj, _ = cv2.projectPoints(all_obj_points[i], rvecs[i], tvecs[i], K, dist)
            err = float(np.sqrt(np.mean((all_img_points[i] - proj.reshape(-1, 2)) ** 2)))
            per_view_errors.append(err)

        return MonoCalibResult(
            rms_reprojection_px=float(ret),
            K=K,
            distortion=dist.ravel(),
            num_views=len(all_obj_points),
            num_corners_total=total_corners,
            per_view_errors=per_view_errors,
        )

    def calibrate_stereo(
        self,
        left_images: list[np.ndarray],
        right_images: list[np.ndarray],
        image_size: tuple[int, int],
        K_left: np.ndarray,
        dist_left: np.ndarray,
        K_right: np.ndarray,
        dist_right: np.ndarray,
        fix_intrinsics: bool = True,
    ) -> StereoCalibResult:
        """Perform stereo extrinsic calibration between synchronized left and right image pairs."""
        if len(left_images) != len(right_images):
            raise ValueError(f"Mismatched stereo pair lengths: {len(left_images)} vs {len(right_images)}")

        obj_paired = []
        img_left_paired = []
        img_right_paired = []

        for im_l, im_r in zip(left_images, right_images, strict=True):
            _, ids_l, obj_l, img_l = self.detect_board(im_l)
            _, ids_r, obj_r, img_r = self.detect_board(im_r)

            if obj_l is not None and obj_r is not None and ids_l is not None and ids_r is not None:
                # Find common corner IDs detected in both left and right frames
                common_ids = np.intersect1d(ids_l.flatten(), ids_r.flatten())
                if len(common_ids) >= 6:
                    mask_l = np.isin(ids_l.flatten(), common_ids)
                    mask_r = np.isin(ids_r.flatten(), common_ids)

                    obj_paired.append(obj_l[mask_l])
                    img_left_paired.append(img_l[mask_l])
                    img_right_paired.append(img_r[mask_r])

        if len(obj_paired) < 1:
            raise ValueError("No common ChArUco corners detected between stereo pairs.")

        flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else 0

        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_paired,
            img_left_paired,
            img_right_paired,
            K_left,
            dist_left,
            K_right,
            dist_right,
            image_size,
            flags=flags,
        )

        baseline_m = float(np.linalg.norm(T))

        return StereoCalibResult(
            rms_stereo_px=float(ret),
            K_left=K_left,
            dist_left=dist_left,
            K_right=K_right,
            dist_right=dist_right,
            R=R,
            T=T,
            baseline_m=baseline_m,
            E=E,
            F=F,
            num_stereo_pairs=len(obj_paired),
        )

