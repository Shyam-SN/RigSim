"""Image content fault injectors with ground-truth label generation."""

from typing import Any

import cv2
import numpy as np

from rigsim.config.camera import CameraIntrinsics
from rigsim.config.transform import Transform
from rigsim.config.types import FaultType
from rigsim.faults.base import BaseFaultInjector


class MotionBlurInjector(BaseFaultInjector):
    """Fault 13: Motion blur from prolonged exposure or fast jitter."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 0.8,
        seed: int = 42,
        kernel_size: int = 21,
        angle_deg: float = 45.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.MOTION_BLUR,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            kernel_size=kernel_size,
            angle_deg=angle_deg,
            **kwargs,
        )
        self.ksize = max(3, int(round(kernel_size * severity)))
        if self.ksize % 2 == 0:
            self.ksize += 1
        self.angle_deg = angle_deg

    def modify_image(self, image_rgb: np.ndarray, timestamp_ns: int) -> np.ndarray:
        if not self.is_active(timestamp_ns):
            return image_rgb

        # Construct directional motion blur kernel
        kernel = np.zeros((self.ksize, self.ksize), dtype=np.float32)
        center = self.ksize // 2
        kernel[center, :] = 1.0

        # Rotate kernel by angle_deg
        rot_mat = cv2.getRotationMatrix2D((center, center), self.angle_deg, 1.0)
        kernel = np.asarray(
            cv2.warpAffine(kernel, rot_mat, (self.ksize, self.ksize)), dtype=np.float32
        )
        k_sum = float(np.sum(kernel))
        if k_sum > 0:
            kernel /= k_sum

        filtered: np.ndarray = cv2.filter2D(image_rgb, -1, kernel)
        return filtered


class ExposureFaultInjector(BaseFaultInjector):
    """Fault 14: Overexposure (blown out) or underexposure (dark)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        mode: str = "overexposure",  # overexposure or underexposure
        gain: float = 3.5,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=(
                FaultType.OVEREXPOSURE if mode == "overexposure" else FaultType.UNDEREXPOSURE
            ),
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            mode=mode,
            gain=gain,
            **kwargs,
        )
        self.mode = mode
        self.gain = (
            gain * severity if mode == "overexposure" else max(0.05, 1.0 - (0.95 * severity))
        )

    def modify_image(self, image_rgb: np.ndarray, timestamp_ns: int) -> np.ndarray:
        if not self.is_active(timestamp_ns):
            return image_rgb

        img_float = image_rgb.astype(np.float32)
        if self.mode == "overexposure":
            img_mod = np.clip(img_float * self.gain + (40 * self.severity), 0.0, 255.0)
        else:
            img_mod = np.clip(img_float * self.gain, 0.0, 255.0)
        return np.asarray(img_mod, dtype=np.uint8)


class LensOcclusionInjector(BaseFaultInjector):
    """Fault 15: Lens occlusion (finger blob or full black obstruction)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 0.9,
        seed: int = 42,
        mode: str = "finger_blob",  # finger_blob or full_black
        coverage: float = 0.7,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.LENS_OCCLUSION,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            mode=mode,
            coverage=coverage,
            **kwargs,
        )
        self.mode = mode
        self.coverage = coverage * severity

    def modify_image(self, image_rgb: np.ndarray, timestamp_ns: int) -> np.ndarray:
        if not self.is_active(timestamp_ns):
            return image_rgb

        h, w = image_rgb.shape[:2]
        out = image_rgb.copy()

        if self.mode == "full_black":
            return np.zeros_like(image_rgb)

        # Draw semi-translucent skin-toned / dark occlusion blob
        mask = np.zeros((h, w), dtype=np.uint8)
        radius = int(min(h, w) * 0.45 * self.coverage)
        center = (int(w * 0.4), int(h * 0.5))
        cv2.circle(mask, center, radius, 255, -1)
        mask_blurred = cv2.GaussianBlur(mask, (51, 51), 25)

        # Alpha blend occlusion
        alpha = (mask_blurred.astype(np.float32) / 255.0)[..., np.newaxis]
        occlusion_color = np.array([30, 25, 20], dtype=np.float32)
        out = np.clip(
            out.astype(np.float32) * (1.0 - 0.95 * alpha) + occlusion_color * alpha, 0, 255
        )
        return np.asarray(out, dtype=np.uint8)


class FrozenFrameInjector(BaseFaultInjector):
    """Fault 16: Frozen video frame while timestamps advance."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.FROZEN_FRAME,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            **kwargs,
        )
        self.frozen_image: np.ndarray | None = None

    def modify_image(self, image_rgb: np.ndarray, timestamp_ns: int) -> np.ndarray:
        if not self.is_active(timestamp_ns):
            return image_rgb

        if self.frozen_image is None:
            self.frozen_image = image_rgb.copy()
        return self.frozen_image.copy()


class CorruptedFrameInjector(BaseFaultInjector):
    """Fault 17: Image corruption (transmission stripes, decode blocks, green/pink artifact)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.CORRUPTED_FRAME,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            **kwargs,
        )

    def modify_image(self, image_rgb: np.ndarray, timestamp_ns: int) -> np.ndarray:
        if not self.is_active(timestamp_ns):
            return image_rgb

        out = image_rgb.copy()
        h, w = out.shape[:2]

        # Inject horizontal corruption bands
        for _ in range(int(3 * self.severity)):
            y1 = self.rng.integers(0, h - 20)
            y2 = min(h, y1 + self.rng.integers(10, 40))
            color = self.rng.integers(0, 256, size=3, dtype=np.uint8)
            out[y1:y2, :] = color

        return out


class HeavyNoiseInjector(BaseFaultInjector):
    """Fault 18: Heavy sensor noise (low light / high analog gain)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        noise_std: float = 35.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.HEAVY_NOISE,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            noise_std=noise_std,
            **kwargs,
        )
        self.noise_std = noise_std * severity

    def modify_image(self, image_rgb: np.ndarray, timestamp_ns: int) -> np.ndarray:
        if not self.is_active(timestamp_ns):
            return image_rgb

        noise = self.rng.normal(0.0, self.noise_std, image_rgb.shape)
        noisy = np.clip(image_rgb.astype(np.float32) + noise, 0.0, 255.0)
        return noisy.astype(np.uint8)


class CameraBumpInjector(BaseFaultInjector):
    """Fault 19: Physical camera bump (sudden extrinsic shift)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        delta_rotation_deg: float = 1.5,
        delta_translation_mm: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.CAMERA_BUMP,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            delta_rotation_deg=delta_rotation_deg,
            delta_translation_mm=delta_translation_mm,
            **kwargs,
        )
        self.rot_deg = delta_rotation_deg * severity
        self.trans_m = (delta_translation_mm / 1000.0) * severity

    def modify_extrinsics(self, transform: Transform, timestamp_ns: int) -> Transform:
        if timestamp_ns < self.t_start_ns:
            return transform

        from scipy.spatial.transform import Rotation

        # Perturb translation and rotation
        tx, ty, tz = transform.translation
        tx += self.trans_m
        tz += self.trans_m * 0.5

        # Small rotation about Y and Z axes
        bump_rot = Rotation.from_euler("yz", [self.rot_deg, self.rot_deg * 0.7], degrees=True)
        orig_rot = Rotation.from_quat(transform.rotation_xyzw)
        new_rot = bump_rot * orig_rot
        new_q = tuple(float(x) for x in new_rot.as_quat())

        return Transform.from_translation_quaternion((tx, ty, tz), new_q)  # type: ignore[arg-type]


class IntrinsicShiftInjector(BaseFaultInjector):
    """Fault 20: Camera focal length / principal point change (lens thermal expansion or bump)."""

    def __init__(
        self,
        stream_id: str,
        t_start_ns: int,
        t_end_ns: int,
        severity: float = 1.0,
        seed: int = 42,
        focal_shift_ratio: float = 0.08,
        **kwargs: Any,
    ):
        super().__init__(
            fault_type=FaultType.INTRINSIC_SHIFT,
            stream_id=stream_id,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            severity=severity,
            seed=seed,
            focal_shift_ratio=focal_shift_ratio,
            **kwargs,
        )
        self.shift_ratio = focal_shift_ratio * severity

    def modify_intrinsics(
        self, intrinsics: CameraIntrinsics, timestamp_ns: int
    ) -> CameraIntrinsics:
        if timestamp_ns < self.t_start_ns:
            return intrinsics

        return CameraIntrinsics(
            fx=intrinsics.fx * (1.0 + self.shift_ratio),
            fy=intrinsics.fy * (1.0 + self.shift_ratio),
            cx=intrinsics.cx + 5.0 * self.severity,
            cy=intrinsics.cy - 3.0 * self.severity,
            width=intrinsics.width,
            height=intrinsics.height,
            fov_deg=intrinsics.fov_deg,
        )
