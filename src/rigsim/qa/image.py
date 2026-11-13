"""Image content QA checks (I1-I6)."""

import cv2
import numpy as np

from rigsim.config.qa import ImageThresholds, QAResult
from rigsim.config.types import QAVerdict


def compute_frame_laplacian_variance(image_rgb: np.ndarray) -> float:
    """Compute standard Laplacian variance sharpness metric."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def check_i1_blur(
    stream_id: str,
    frames_rgb: list[np.ndarray],
    timestamps_ns: list[int],
    thresholds: ImageThresholds,
) -> QAResult:
    """I1: Normalized Laplacian variance sharpness check."""
    if len(frames_rgb) < 3:
        return QAResult(
            check_id="I1_BLUR",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="min_normalized_blur_ratio",
            value=1.0,
            threshold=thresholds.min_normalized_blur_ratio,
            explanation="Insufficient frames for blur analysis.",
        )

    variances = np.array(
        [compute_frame_laplacian_variance(img) for img in frames_rgb], dtype=np.float64
    )

    # Compute rolling median (window = 15 frames or len/3)
    k = min(15, len(variances))
    if k % 2 == 0:
        k -= 1
    k = max(3, k)

    # Pad variances for rolling median
    pad_w = k // 2
    padded = np.pad(variances, pad_w, mode="edge")
    rolling_medians = np.array([np.median(padded[i : i + k]) for i in range(len(variances))])
    # Avoid zero division
    rolling_medians = np.maximum(rolling_medians, 1.0)
    normalized_ratios = variances / rolling_medians

    flagged_indices = np.where(normalized_ratios < thresholds.min_normalized_blur_ratio)[0]
    flagged_windows = []
    for idx in flagged_indices:
        flagged_windows.append({"t_start_ns": timestamps_ns[idx], "t_end_ns": timestamps_ns[idx]})

    min_ratio = float(np.min(normalized_ratios))
    blur_drop_fraction = len(flagged_indices) / len(variances)

    verdict = QAVerdict.PASS
    if blur_drop_fraction > 0.05 or min_ratio < (thresholds.min_normalized_blur_ratio * 0.7):
        verdict = QAVerdict.FAIL
        explanation = f"FAIL: Motion blur detected on {len(flagged_indices)} frames (min sharpness ratio {min_ratio:.2f} of rolling median)."
    else:
        explanation = f"Sharp focus verified: min normalized sharpness ratio {min_ratio:.2f} (threshold {thresholds.min_normalized_blur_ratio})."

    return QAResult(
        check_id="I1_BLUR",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="min_normalized_blur_ratio",
        value=min_ratio,
        threshold=thresholds.min_normalized_blur_ratio,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"min_ratio": min_ratio, "flagged_count": len(flagged_indices)},
        explanation=explanation,
    )


def check_i2_exposure(
    stream_id: str,
    frames_rgb: list[np.ndarray],
    timestamps_ns: list[int],
    thresholds: ImageThresholds,
) -> QAResult:
    """I2: Exposure check (underexposure, overexposure, clipping ratios)."""
    if len(frames_rgb) == 0:
        return QAResult(
            check_id="I2_EXPOSURE",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="mean_brightness",
            value=128.0,
            threshold=thresholds.min_mean_brightness,
            explanation="No frames to check.",
        )

    mean_brightnesses = []
    clipping_ratios = []
    flagged_windows = []

    for idx, img in enumerate(frames_rgb):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        mean_b = float(np.mean(gray))
        clip_0 = np.sum(gray == 0)
        clip_255 = np.sum(gray == 255)
        clip_ratio = float((clip_0 + clip_255) / gray.size)

        mean_brightnesses.append(mean_b)
        clipping_ratios.append(clip_ratio)

        if (
            mean_b < thresholds.min_mean_brightness
            or mean_b > thresholds.max_mean_brightness
            or clip_ratio > thresholds.max_clipping_ratio
        ):
            flagged_windows.append(
                {"t_start_ns": timestamps_ns[idx], "t_end_ns": timestamps_ns[idx]}
            )

    min_b = float(np.min(mean_brightnesses))
    max_b = float(np.max(mean_brightnesses))
    max_clip = float(np.max(clipping_ratios))
    bad_ratio = len(flagged_windows) / len(frames_rgb)

    verdict = QAVerdict.PASS
    if bad_ratio > 0.05:
        if (
            min_b < thresholds.min_mean_brightness
            or max_b > thresholds.max_mean_brightness
            or max_clip > 0.35
        ):
            verdict = QAVerdict.FAIL
            explanation = f"FAIL: Severe exposure anomaly on {len(flagged_windows)} frames (mean brightness range [{min_b:.1f}, {max_b:.1f}], max clipping {max_clip * 100:.1f}%)."
        else:
            verdict = QAVerdict.WARN
            explanation = f"WARN: High contrast clipping on {len(flagged_windows)} frames (clipping {max_clip * 100:.1f}% > limit {thresholds.max_clipping_ratio * 100:.0f}%, mean brightness [{min_b:.1f}, {max_b:.1f}])."
    else:
        explanation = f"Balanced exposure: mean brightness range [{min_b:.1f}, {max_b:.1f}], max clipping {max_clip * 100:.1f}%."

    return QAResult(
        check_id="I2_EXPOSURE",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="mean_brightness",
        value=min_b if min_b < thresholds.min_mean_brightness else max_b,
        threshold=thresholds.min_mean_brightness,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"min_brightness": min_b, "max_brightness": max_b, "max_clipping_ratio": max_clip},
        explanation=explanation,
    )


def check_i3_occlusion(
    stream_id: str,
    frames_rgb: list[np.ndarray],
    timestamps_ns: list[int],
    thresholds: ImageThresholds,
) -> QAResult:
    """I3: Lens occlusion / blocked camera detection."""
    flagged_windows = []

    for idx, img in enumerate(frames_rgb):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        # Spatial gradient / edge density
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        flat_fraction = float(np.mean(grad_mag < 3.0))
        mean_grad = float(np.mean(grad_mag))

        # Occlusion: either extreme unfeatured flatness (>80% flat) or high flatness with collapsed edge density
        if flat_fraction > 0.80 or (
            flat_fraction > thresholds.occlusion_min_uniform_ratio and mean_grad < 6.0
        ):
            flagged_windows.append(
                {"t_start_ns": timestamps_ns[idx], "t_end_ns": timestamps_ns[idx]}
            )

    bad_ratio = len(flagged_windows) / max(1, len(frames_rgb))
    verdict = QAVerdict.FAIL if bad_ratio > 0.05 else QAVerdict.PASS

    if verdict == QAVerdict.FAIL:
        explanation = f"FAIL: Lens occlusion detected on {len(flagged_windows)} frames (uniform obstruction > {thresholds.occlusion_min_uniform_ratio * 100:.0f}%)."
    else:
        explanation = "Clear lens field-of-view verified (no persistent occlusions)."

    return QAResult(
        check_id="I3_OCCLUSION",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="occluded_frames_ratio",
        value=float(bad_ratio),
        threshold=0.05,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"occluded_frames_count": len(flagged_windows)},
        explanation=explanation,
    )


def check_i4_frozen_frames(
    stream_id: str,
    frames_rgb: list[np.ndarray],
    timestamps_ns: list[int],
    thresholds: ImageThresholds,
) -> QAResult:
    """I4: Frozen frame detection (identical images with moving timestamps)."""
    if len(frames_rgb) < 2:
        return QAResult(
            check_id="I4_FROZEN_FRAMES",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="max_frozen_run",
            value=0.0,
            threshold=thresholds.max_frozen_frame_count,
            explanation="Insufficient frames for frozen check.",
        )

    current_frozen_run = 0
    max_frozen_run = 0
    flagged_windows = []

    for i in range(1, len(frames_rgb)):
        # Mean absolute pixel difference
        diff = float(
            np.mean(np.abs(frames_rgb[i].astype(np.float32) - frames_rgb[i - 1].astype(np.float32)))
        )
        if diff < 0.01:
            current_frozen_run += 1
            if current_frozen_run >= thresholds.max_frozen_frame_count:
                flagged_windows.append(
                    {"t_start_ns": timestamps_ns[i - 1], "t_end_ns": timestamps_ns[i]}
                )
            if current_frozen_run > max_frozen_run:
                max_frozen_run = current_frozen_run
        else:
            current_frozen_run = 0

    verdict = (
        QAVerdict.FAIL if max_frozen_run >= thresholds.max_frozen_frame_count else QAVerdict.PASS
    )
    if verdict == QAVerdict.FAIL:
        explanation = f"FAIL: Frozen frames detected (run of {max_frozen_run + 1} identical images with advancing timestamps)."
    else:
        explanation = f"Dynamic video verified (max identical run {max_frozen_run} frames, limit {thresholds.max_frozen_frame_count})."

    return QAResult(
        check_id="I4_FROZEN_FRAMES",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="max_frozen_run",
        value=float(max_frozen_run),
        threshold=float(thresholds.max_frozen_frame_count),
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"max_frozen_run": max_frozen_run},
        explanation=explanation,
    )


def check_i5_corruption(
    stream_id: str,
    frames_rgb: list[np.ndarray],
    timestamps_ns: list[int],
) -> QAResult:
    """I5: Corrupted frames, transmission stripes, decode failures."""
    flagged_windows = []

    for idx, img in enumerate(frames_rgb):
        if img.size == 0 or np.mean(img) < 1.0 or np.mean(img) > 254.0:
            # Decode failure / all-black or all-white dropout
            flagged_windows.append(
                {"t_start_ns": timestamps_ns[idx], "t_end_ns": timestamps_ns[idx]}
            )
            continue

        # Horizontal transmission stripe / packet drop detection:
        # A dropped network packet or macroblock row fills scanlines with solid color.
        # This causes both an abrupt step in row mean AND near-zero intra-row standard deviation across adjacent stripe scanlines.
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        row_means = np.mean(gray, axis=1)
        row_stds = np.std(gray, axis=1)
        row_diffs = np.abs(np.diff(row_means))

        stripe_detected = False
        for r in range(len(row_diffs)):
            if row_diffs[r] > 35.0:
                if (row_stds[r] < 1.5 and r > 0 and row_stds[r - 1] < 1.5) or (
                    r + 2 < len(row_stds) and row_stds[r + 1] < 1.5 and row_stds[r + 2] < 1.5
                ):
                    stripe_detected = True
                    break

        # Check for extreme chroma decode corruption (e.g. green/pink YUV decode fallback)
        if not stripe_detected and img.ndim == 3:
            r_ch = img[..., 0].astype(np.int32)
            g_ch = img[..., 1].astype(np.int32)
            b_ch = img[..., 2].astype(np.int32)
            green_pixels = np.sum((g_ch > 220) & (r_ch < 40) & (b_ch < 40))
            magenta_pixels = np.sum((r_ch > 220) & (b_ch > 220) & (g_ch < 40))
            if (green_pixels + magenta_pixels) / img.size > 0.05:
                stripe_detected = True

        if stripe_detected:
            flagged_windows.append(
                {"t_start_ns": timestamps_ns[idx], "t_end_ns": timestamps_ns[idx]}
            )

    corrupted_count = len(flagged_windows)
    verdict = QAVerdict.FAIL if corrupted_count > 0 else QAVerdict.PASS

    if verdict == QAVerdict.FAIL:
        explanation = f"FAIL: Detected {corrupted_count} corrupted / stripe artifact frames."
    else:
        explanation = "Zero corruption or decode failure artifacts detected."

    return QAResult(
        check_id="I5_CORRUPTION",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="corrupted_count",
        value=float(corrupted_count),
        threshold=0.0,
        flagged_windows=[
            {"t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"]} for w in flagged_windows
        ],  # type: ignore[arg-type]
        evidence={"corrupted_count": corrupted_count},
        explanation=explanation,
    )


def check_i6_noise_level(
    stream_id: str,
    frames_rgb: list[np.ndarray],
    timestamps_ns: list[int],
) -> QAResult:
    """I6: High-frequency sensor noise estimation."""
    if len(frames_rgb) == 0:
        return QAResult(
            check_id="I6_NOISE_LEVEL",
            stream_id=stream_id,
            verdict=QAVerdict.PASS,
            metric_name="estimated_noise_sigma",
            value=0.0,
            threshold=25.0,
            explanation="No frames to evaluate.",
        )

    # Estimate noise standard deviation using Immerkaer's fast noise estimation method
    # N = sum(|I * [1 -2 1; -2 4 -2; 1 -2 1]|) * sqrt(pi / 2) / (6 * (W-2) * (H-2))
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)

    noise_estimates = []
    for img in frames_rgb:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        h, w = gray.shape
        conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sigma = float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))
        noise_estimates.append(sigma)

    mean_noise_sigma = float(np.mean(noise_estimates))
    threshold_sigma = 25.0

    verdict = QAVerdict.WARN if mean_noise_sigma > threshold_sigma else QAVerdict.PASS
    if verdict == QAVerdict.WARN:
        explanation = f"Elevated sensor noise level (estimated noise sigma {mean_noise_sigma:.1f} > threshold {threshold_sigma})."
    else:
        explanation = f"Clean sensor noise profile (estimated sigma {mean_noise_sigma:.1f})."

    return QAResult(
        check_id="I6_NOISE_LEVEL",
        stream_id=stream_id,
        verdict=verdict,
        metric_name="estimated_noise_sigma",
        value=mean_noise_sigma,
        threshold=threshold_sigma,
        evidence={"mean_sigma": mean_noise_sigma},
        explanation=explanation,
    )
