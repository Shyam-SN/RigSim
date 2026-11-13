"""Unit tests for image QA checks (I1, I2, I3, I4, I5, I6)."""

import cv2
import numpy as np
import pytest

from rigsim.config.qa import ImageThresholds, QAVerdict
from rigsim.faults.image import (
    CorruptedFrameInjector,
    ExposureFaultInjector,
    FrozenFrameInjector,
    HeavyNoiseInjector,
    LensOcclusionInjector,
    MotionBlurInjector,
)
from rigsim.qa.image import (
    check_i1_blur,
    check_i2_exposure,
    check_i3_occlusion,
    check_i4_frozen_frames,
    check_i5_corruption,
    check_i6_noise_level,
)


@pytest.fixture
def default_thresholds() -> ImageThresholds:
    return ImageThresholds(
        min_normalized_blur_ratio=0.50,
        min_mean_brightness=20.0,
        max_mean_brightness=235.0,
        max_clipping_ratio=0.10,
        max_frozen_frame_count=2,
        max_corrupted_frame_count=0,
        occlusion_min_uniform_ratio=0.60,
    )


@pytest.fixture
def synthetic_scene_image() -> np.ndarray:
    """Generate a rich synthetic image with gradients, shapes, and textures."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    # Background gradient
    for y in range(240):
        img[y, :, :] = int(60 + 0.4 * y)
    # Shapes
    cv2.rectangle(img, (40, 40), (120, 140), (200, 50, 50), -1)
    cv2.circle(img, (200, 100), 40, (50, 200, 50), -1)
    cv2.putText(img, "RigSim QA Test", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    return img


def test_i1_blur_clean_and_defocused(
    default_thresholds: ImageThresholds, synthetic_scene_image: np.ndarray
) -> None:
    frames = [synthetic_scene_image.copy() for _ in range(20)]
    ts = [1_000_000_000 + i * 33_333_333 for i in range(20)]

    # Clean
    res_clean = check_i1_blur("cam0", frames, ts, default_thresholds)
    assert res_clean.verdict == QAVerdict.PASS

    # Defocused frames 5..10
    blur_injector = MotionBlurInjector("cam0", ts[5], ts[10], severity=1.0, kernel_size=25)
    blurred_frames = [blur_injector.modify_image(f, t) for f, t in zip(frames, ts, strict=False)]
    res_blurred = check_i1_blur("cam0", blurred_frames, ts, default_thresholds)
    assert res_blurred.verdict == QAVerdict.FAIL
    assert len(res_blurred.flagged_windows) >= 1


def test_i2_exposure_variations(
    default_thresholds: ImageThresholds, synthetic_scene_image: np.ndarray
) -> None:
    frames = [synthetic_scene_image.copy() for _ in range(10)]
    ts = [1_000_000_000 + i * 33_333_333 for i in range(10)]

    # Clean
    assert check_i2_exposure("cam0", frames, ts, default_thresholds).verdict == QAVerdict.PASS

    # Underexposure
    under_inj = ExposureFaultInjector("cam0", ts[0], ts[-1], severity=1.0, mode="underexposure", gain=0.05)
    under_frames = [under_inj.modify_image(f, t) for f, t in zip(frames, ts, strict=False)]
    res_under = check_i2_exposure("cam0", under_frames, ts, default_thresholds)
    assert res_under.verdict == QAVerdict.FAIL

    # Overexposure
    over_inj = ExposureFaultInjector("cam0", ts[0], ts[-1], severity=1.0, mode="overexposure", gain=4.0)
    over_frames = [over_inj.modify_image(f, t) for f, t in zip(frames, ts, strict=False)]
    res_over = check_i2_exposure("cam0", over_frames, ts, default_thresholds)
    assert res_over.verdict == QAVerdict.FAIL


def test_i3_occlusion_detection(
    default_thresholds: ImageThresholds, synthetic_scene_image: np.ndarray
) -> None:
    frames = [synthetic_scene_image.copy() for _ in range(10)]
    ts = [1_000_000_000 + i * 33_333_333 for i in range(10)]

    # Clean
    assert check_i3_occlusion("cam0", frames, ts, default_thresholds).verdict == QAVerdict.PASS

    # Fully occluded
    occ_inj = LensOcclusionInjector("cam0", ts[0], ts[-1], severity=1.0, mode="full_black")
    occ_frames = [occ_inj.modify_image(f, t) for f, t in zip(frames, ts, strict=False)]
    res_occ = check_i3_occlusion("cam0", occ_frames, ts, default_thresholds)
    assert res_occ.verdict == QAVerdict.FAIL


def test_i4_frozen_frames_detection(
    default_thresholds: ImageThresholds, synthetic_scene_image: np.ndarray
) -> None:
    ts = [1_000_000_000 + i * 33_333_333 for i in range(15)]
    # Moving frames (slightly changing pixel values)
    moving_frames = []
    for i in range(15):
        f = synthetic_scene_image.copy()
        cv2.circle(f, (50 + i * 5, 50), 10, (255, 255, 255), -1)
        moving_frames.append(f)

    res_moving = check_i4_frozen_frames("cam0", moving_frames, ts, default_thresholds)
    assert res_moving.verdict == QAVerdict.PASS

    # Freeze frames from index 3 to 10 (7 consecutive identical frames)
    freeze_inj = FrozenFrameInjector("cam0", ts[3], ts[10], severity=1.0)
    frozen_frames = [freeze_inj.modify_image(f, t) for f, t in zip(moving_frames, ts, strict=False)]
    res_frozen = check_i4_frozen_frames("cam0", frozen_frames, ts, default_thresholds)
    assert res_frozen.verdict == QAVerdict.FAIL
    assert res_frozen.value >= 3


def test_i5_corruption_detection(synthetic_scene_image: np.ndarray) -> None:
    frames = [synthetic_scene_image.copy() for _ in range(10)]
    ts = [1_000_000_000 + i * 33_333_333 for i in range(10)]

    # Clean
    assert check_i5_corruption("cam0", frames, ts).verdict == QAVerdict.PASS

    # Inject horizontal stripe corruption
    corrupt_inj = CorruptedFrameInjector("cam0", ts[3], ts[5], severity=1.0)
    corrupted_frames = [corrupt_inj.modify_image(f, t) for f, t in zip(frames, ts, strict=False)]
    res_corrupt = check_i5_corruption("cam0", corrupted_frames, ts)
    assert res_corrupt.verdict == QAVerdict.FAIL
    assert res_corrupt.value >= 1


def test_i6_noise_level_detection(synthetic_scene_image: np.ndarray) -> None:
    frames = [synthetic_scene_image.copy() for _ in range(10)]
    ts = [1_000_000_000 + i * 33_333_333 for i in range(10)]

    # Clean
    res_clean = check_i6_noise_level("cam0", frames, ts)
    assert res_clean.verdict == QAVerdict.PASS
    assert res_clean.value < 10.0

    # Heavy sensor noise (sigma = 40.0)
    noise_inj = HeavyNoiseInjector("cam0", ts[0], ts[-1], severity=1.0, noise_std=40.0)
    noisy_frames = [noise_inj.modify_image(f, t) for f, t in zip(frames, ts, strict=False)]
    res_noise = check_i6_noise_level("cam0", noisy_frames, ts)
    assert res_noise.verdict == QAVerdict.WARN
    assert res_noise.value > 25.0
