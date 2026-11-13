"""IMU-Camera temporal offset estimation via visual angular velocity and gyro cross-correlation."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import interp1d


@dataclass
class ImuCamOffsetResult:
    offset_ms: float
    correlation_peak: float
    time_lags_ms: np.ndarray
    correlations: np.ndarray
    explanation: str


class ImuCameraSyncEstimator:
    """Estimates the hardware time offset between a camera and an IMU.

    Cross-correlates visual angular velocity (from sparse optical flow)
    with gyroscope angular velocity norm, followed by sub-frame parabolic peak refinement.
    """

    def __init__(
        self,
        max_lag_ms: float = 100.0,
        grid_dt_s: float = 0.001,  # 1 kHz uniform grid for 1 ms resolution before interpolation
        min_features: int = 15,
    ):
        self.max_lag_ms = max_lag_ms
        self.grid_dt_s = grid_dt_s
        self.min_features = min_features

    def estimate_offset(
        self,
        frames_rgb: list[np.ndarray],
        cam_timestamps_ns: list[int],
        gyro_readings: list[list[float]] | np.ndarray,  # shape (N, 3) in rad/s
        imu_timestamps_ns: list[int],
        K: np.ndarray | None = None,
    ) -> ImuCamOffsetResult:
        if len(frames_rgb) < 10 or len(imu_timestamps_ns) < 20:
            return ImuCamOffsetResult(
                offset_ms=0.0,
                correlation_peak=0.0,
                time_lags_ms=np.array([0.0]),
                correlations=np.array([0.0]),
                explanation="Insufficient visual or IMU samples for cross-correlation.",
            )

        # 1. Compute visual angular velocity from consecutive frames
        vis_times_s = []
        vis_rates = []

        h, w = frames_rgb[0].shape[:2]
        if K is None:
            # Fallback pinhole intrinsics
            f = max(h, w)
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)

        prev_gray = (
            cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
            if frames_rgb[0].ndim == 3
            else frames_rgb[0]
        )

        for i in range(1, len(frames_rgb)):
            curr_gray = (
                cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
                if frames_rgb[i].ndim == 3
                else frames_rgb[i]
            )
            dt_s = (cam_timestamps_ns[i] - cam_timestamps_ns[i - 1]) / 1e9
            if dt_s <= 0:
                prev_gray = curr_gray
                continue

            # Detect Shi-Tomasi corners
            pts_prev = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10
            )

            omega_mag = 0.0
            if pts_prev is not None and len(pts_prev) >= self.min_features:
                # Track using Lucas-Kanade optical flow
                pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                    prev_gray, curr_gray, pts_prev, None, winSize=(21, 21), maxLevel=3
                )
                valid = status.flatten() == 1
                p0 = pts_prev[valid]
                p1 = pts_curr[valid]

                if len(p0) >= self.min_features:
                    # Robust homography with SVD polar decomposition for rotation extraction
                    H, _ = cv2.findHomography(p0.reshape(-1, 2), p1.reshape(-1, 2), cv2.RANSAC, 2.0)
                    if H is not None:
                        K_inv = np.linalg.inv(K)
                        H_norm = K_inv @ H @ K
                        U, _, Vt = np.linalg.svd(H_norm)
                        R = U @ Vt
                        if np.linalg.det(R) < 0:
                            R = -R
                        tr = np.trace(R)
                        cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
                        omega_mag = float(np.arccos(cos_theta) / dt_s)
                    else:
                        # Fallback: mean optical flow displacement normalized by focal length
                        flow = p1 - p0
                        mean_disp_px = float(np.mean(np.linalg.norm(flow, axis=1)))
                        omega_mag = float((mean_disp_px / K[0, 0]) / dt_s)

            mid_t = (cam_timestamps_ns[i] + cam_timestamps_ns[i - 1]) / 2.0 / 1e9
            vis_times_s.append(mid_t)
            vis_rates.append(omega_mag)
            prev_gray = curr_gray

        # 2. Extract IMU gyro angular rate norm
        gyro_arr = np.array(gyro_readings, dtype=np.float64)
        gyro_norm = np.linalg.norm(gyro_arr, axis=1)
        imu_times_s = np.array(imu_timestamps_ns, dtype=np.float64) / 1e9

        # 3. Find common temporal overlap
        t_min = max(vis_times_s[0], imu_times_s[0])
        t_max = min(vis_times_s[-1], imu_times_s[-1])

        if (t_max - t_min) < 1.0:
            return ImuCamOffsetResult(
                offset_ms=0.0,
                correlation_peak=0.0,
                time_lags_ms=np.array([0.0]),
                correlations=np.array([0.0]),
                explanation="Overlap between visual motion and IMU too short (<1s).",
            )

        # 4. Resample onto uniform grid
        grid_t = np.arange(t_min, t_max, self.grid_dt_s)
        f_vis = interp1d(vis_times_s, vis_rates, kind="linear", fill_value="extrapolate")
        f_imu = interp1d(imu_times_s, gyro_norm, kind="linear", fill_value="extrapolate")

        sig_vis = f_vis(grid_t)
        sig_imu = f_imu(grid_t)

        # Zero-mean normalize
        std_vis = np.std(sig_vis)
        std_imu = np.std(sig_imu)

        if std_vis < 1e-4 or std_imu < 1e-4:
            # Low excitation motion
            return ImuCamOffsetResult(
                offset_ms=0.0,
                correlation_peak=0.0,
                time_lags_ms=np.array([0.0]),
                correlations=np.array([0.0]),
                explanation="Insufficient rotational excitation for correlation (near static).",
            )

        sig_vis_norm = (sig_vis - np.mean(sig_vis)) / std_vis
        sig_imu_norm = (sig_imu - np.mean(sig_imu)) / std_imu

        # 5. Cross-correlation over [-max_lag, +max_lag]
        max_lag_samples = int(round(self.max_lag_ms / 1000.0 / self.grid_dt_s))
        lags = np.arange(-max_lag_samples, max_lag_samples + 1)
        corr = np.correlate(sig_vis_norm, sig_imu_norm, mode="full")
        center = len(sig_imu_norm) - 1
        corr_slice = corr[center - max_lag_samples : center + max_lag_samples + 1] / len(grid_t)

        # 6. Peak search with MAP prior (disambiguate distinct periodic local maxima within 3% of max)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(corr_slice)
        if len(peaks) > 0:
            max_val = float(np.max(corr_slice[peaks]))
            competitive = [p for p in peaks if corr_slice[p] >= max_val - 0.03]
            if len(competitive) > 1:
                peak_idx = int(competitive[np.argmin(np.abs(lags[competitive]))])
            else:
                peak_idx = int(peaks[np.argmax(corr_slice[peaks])])
        else:
            peak_idx = int(np.argmax(corr_slice))

        peak_lag = lags[peak_idx]
        peak_val = float(corr_slice[peak_idx])

        # Parabolic refinement
        refined_lag = float(peak_lag)
        if 0 < peak_idx < len(corr_slice) - 1:
            y_neg = corr_slice[peak_idx - 1]
            y_zero = corr_slice[peak_idx]
            y_pos = corr_slice[peak_idx + 1]
            denom = y_neg - 2.0 * y_zero + y_pos
            if abs(denom) > 1e-6:
                delta = (y_neg - y_pos) / (2.0 * denom)
                if abs(delta) < 1.0:
                    refined_lag = peak_lag + delta

        offset_s = refined_lag * self.grid_dt_s
        offset_ms = float(offset_s * 1000.0)

        explanation = (
            f"Estimated IMU-Camera offset: {offset_ms:+.2f} ms "
            f"(cross-correlation peak r = {peak_val:.3f})."
        )

        return ImuCamOffsetResult(
            offset_ms=offset_ms,
            correlation_peak=peak_val,
            time_lags_ms=lags * self.grid_dt_s * 1000.0,
            correlations=corr_slice,
            explanation=explanation,
        )
