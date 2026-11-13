"""Cross-device optical clock synchronization using screen-displayed QR clocks."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from rigsim.config.clock import ClockModel


@dataclass
class QrClockSyncResult:
    offset_ms: float  # Initial clock offset at start of recording (mac - phone) in ms
    drift_ppm: float  # Relative clock frequency drift (parts-per-million)
    slope_a: float  # Fitted linear model slope: t_mac = a * t_phone + b
    intercept_b_s: float  # Intercept in seconds relative to base
    residual_jitter_std_ms: float  # Residual jitter standard deviation (1-sigma)
    residual_jitter_p99_ms: float  # 99th percentile residual error
    decoded_count: int  # Number of valid decoded frames
    total_frames_checked: int  # Total candidate video frames evaluated
    decode_success_rate: float  # Ratio of valid QR decodes
    error_budget_ms: dict[str, float]  # Quantization and readout error breakdown
    fitted_clock_model: ClockModel  # ClockModel instance for pipeline synchronization
    explanation: str


class QrClockSyncEstimator:
    """Estimates cross-device clock offset and frequency drift from filmed QR clock video."""

    def __init__(
        self,
        display_refresh_rate_hz: float = 60.0,
        rolling_shutter_readout_ms: float = 12.0,
    ):
        self.display_refresh_rate_hz = display_refresh_rate_hz
        self.rolling_shutter_readout_ms = rolling_shutter_readout_ms
        self.detector = cv2.QRCodeDetector()

    def process_frames(
        self,
        frames: list[np.ndarray],
        phone_timestamps_ns: list[int],
    ) -> QrClockSyncResult:
        """Decode QR codes in each frame and fit robust linear time mapping."""
        if len(frames) != len(phone_timestamps_ns):
            raise ValueError(
                f"Mismatch: {len(frames)} frames vs {len(phone_timestamps_ns)} timestamps"
            )

        pairs: list[tuple[int, int]] = []  # (phone_time_ns, mac_time_ns)

        for idx, frame in enumerate(frames):
            phone_ts = phone_timestamps_ns[idx]
            mac_ts = self._decode_qr_timestamp(frame)
            if mac_ts is not None:
                pairs.append((phone_ts, mac_ts))

        return self.fit_clock_model(pairs, total_checked=len(frames))

    def process_video_file(
        self,
        video_path: Path,
        phone_timestamps_ns: list[int] | None = None,
        max_frames: int | None = None,
    ) -> QrClockSyncResult:
        """Decode QR video file directly and fit robust clock sync model."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        frames_checked = 0
        pairs: list[tuple[int, int]] = []

        cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frames_checked >= max_frames):
                break

            frames_checked += 1
            if phone_timestamps_ns and frame_idx < len(phone_timestamps_ns):
                phone_ts = phone_timestamps_ns[frame_idx]
            else:
                # Use video elapsed timestamp in nanoseconds
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                phone_ts = int(pos_msec * 1_000_000)

            mac_ts = self._decode_qr_timestamp(frame)
            if mac_ts is not None:
                pairs.append((phone_ts, mac_ts))

            frame_idx += 1

        cap.release()
        return self.fit_clock_model(pairs, total_checked=frames_checked)

    def _decode_qr_timestamp(self, frame: np.ndarray) -> int | None:
        """Decode single QR code and parse Mac epoch timestamp in nanoseconds."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        val, _, _ = self.detector.detectAndDecode(gray)
        if not val:
            # Try contrast enhancement if lighting was poor
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            val, _, _ = self.detector.detectAndDecode(enhanced)

        if not val:
            return None

        # Parse payload
        try:
            if val.startswith("{"):
                payload = json.loads(val)
                return int(payload["t_ns"])
            elif val.isdigit():
                return int(val)
        except Exception:
            return None

        return None

    def fit_clock_model(
        self,
        timestamp_pairs: list[tuple[int, int]],
        total_checked: int | None = None,
    ) -> QrClockSyncResult:
        """Fit robust Huber regression model: t_mac = a * t_phone + b."""
        n_pairs = len(timestamp_pairs)
        total_frames = total_checked if total_checked is not None else n_pairs
        decode_rate = (n_pairs / max(1, total_frames)) if total_frames > 0 else 0.0

        if n_pairs < 5:
            # Insufficient decodes for robust line fit
            return QrClockSyncResult(
                offset_ms=0.0,
                drift_ppm=0.0,
                slope_a=1.0,
                intercept_b_s=0.0,
                residual_jitter_std_ms=0.0,
                residual_jitter_p99_ms=0.0,
                decoded_count=n_pairs,
                total_frames_checked=total_frames,
                decode_success_rate=decode_rate,
                error_budget_ms={},
                fitted_clock_model=ClockModel(
                    clock_id="phone_synced_clock",
                    offset_ns=0,
                    drift_ppm=0.0,
                    jitter_std_ns=0,
                ),
                explanation=f"Insufficient QR code decodes ({n_pairs} < 5) to fit clock sync.",
            )

        phone_arr = np.array([p[0] for p in timestamp_pairs], dtype=np.float64)
        mac_arr = np.array([p[1] for p in timestamp_pairs], dtype=np.float64)

        # Baseline offset subtraction for numerical stability
        t0_phone = phone_arr[0]
        t0_mac = mac_arr[0]

        x_s = (phone_arr - t0_phone) / 1e9
        y_s = (mac_arr - t0_mac) / 1e9

        # Huber loss robust regression: minimize ||y - (a*x + b)||_Huber
        def huber_residuals(params: list[float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.asarray((params[0] * x + params[1]) - y, dtype=np.float64)

        res = least_squares(
            huber_residuals,
            x0=[1.0, 0.0],
            loss="huber",
            f_scale=0.005,  # 5 ms inlier threshold scale
            args=(x_s, y_s),
        )

        a_slope = float(res.x[0])
        b_intercept_s = float(res.x[1])

        # Clock drift in parts-per-million (PPM)
        drift_ppm = float((a_slope - 1.0) * 1e6)

        # Initial offset at recording start (mac - phone in ms)
        initial_offset_ms = float(((t0_mac - t0_phone) + b_intercept_s * 1e9) / 1e6)

        # Residuals in milliseconds
        fitted_y_s = a_slope * x_s + b_intercept_s
        residuals_ms = (y_s - fitted_y_s) * 1000.0

        jitter_std_ms = float(np.std(residuals_ms))
        jitter_p99_ms = float(np.percentile(np.abs(residuals_ms), 99))

        # Error budget calculation
        disp_quant_ms = (1000.0 / self.display_refresh_rate_hz) / 2.0
        exp_time_est_ms = 16.6 / 2.0  # Approx exposure centroid uncertainty at 30 fps
        rs_skew_ms = self.rolling_shutter_readout_ms / 2.0
        total_theoretical_uncert_ms = float(
            np.sqrt(disp_quant_ms**2 + exp_time_est_ms**2 + rs_skew_ms**2)
        )

        error_budget = {
            "display_quantization_ms": disp_quant_ms,
            "exposure_centroid_ms": exp_time_est_ms,
            "rolling_shutter_skew_ms": rs_skew_ms,
            "theoretical_combined_uncertainty_ms": total_theoretical_uncert_ms,
            "measured_jitter_std_ms": jitter_std_ms,
        }

        fitted_clock = ClockModel(
            clock_id="phone_synced_to_mac",
            offset_ns=int(initial_offset_ms * 1_000_000),
            drift_ppm=drift_ppm,
            jitter_std_ns=int(jitter_std_ms * 1_000_000),
        )

        explanation = (
            f"Fitted clock model across {n_pairs} QR detections ({decode_rate * 100:.1f}% decode rate): "
            f"initial offset {initial_offset_ms:+.2f} ms, drift {drift_ppm:+.2f} ppm, "
            f"residual jitter std {jitter_std_ms:.2f} ms (p99 {jitter_p99_ms:.2f} ms)."
        )

        return QrClockSyncResult(
            offset_ms=initial_offset_ms,
            drift_ppm=drift_ppm,
            slope_a=a_slope,
            intercept_b_s=b_intercept_s,
            residual_jitter_std_ms=jitter_std_ms,
            residual_jitter_p99_ms=jitter_p99_ms,
            decoded_count=n_pairs,
            total_frames_checked=total_frames,
            decode_success_rate=decode_rate,
            error_budget_ms=error_budget,
            fitted_clock_model=fitted_clock,
            explanation=explanation,
        )
