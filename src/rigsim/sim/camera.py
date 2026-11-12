"""Camera model, optics, lens distortion, and physical motion blur rendering."""

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from rigsim.config.camera import CameraDistortion, CameraIntrinsics, CameraSpec
from rigsim.config.transform import Transform


@dataclass
class CameraFrame:
    stream_id: str
    device_time_ns: int
    host_time_ns: int
    image_rgb: np.ndarray  # HxWx3 uint8
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    extrinsics_cv: (
        Transform  # T_world_cam in OpenCV optical conventions (X-right, Y-down, Z-forward)
    )


class CameraSimulator:
    """Simulates a camera: optics, intrinsics computation, distortion, motion blur, and noise."""

    def __init__(self, spec: CameraSpec, model: mujoco.MjModel, seed: int = 42):
        self.spec = spec
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec.stream_id)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{spec.stream_id}' not found in MuJoCo model!")

        self.width, self.height = spec.resolution
        self.exposure_time_s = spec.exposure_time_ms / 1000.0
        self.rng = np.random.default_rng(seed)

        # Compute ground truth OpenCV intrinsics from MuJoCo camera fovy
        fovy_rad = np.deg2rad(float(self.model.cam_fovy[self.cam_id]))
        fy = (self.height / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy  # Square pixels
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.intrinsics = CameraIntrinsics(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=self.width,
            height=self.height,
            fov_deg=float(np.rad2deg(fovy_rad)),
        )

        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # Precompute distortion maps if distortion is enabled
        self.distortion = spec.distortion
        self.dist_coeffs = np.array(spec.distortion.coeffs, dtype=np.float64)
        if len(self.dist_coeffs) > 0 and np.any(np.abs(self.dist_coeffs) > 1e-6):
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.K, self.dist_coeffs, None, self.K, (self.width, self.height), cv2.CV_32FC1
            )
            self.has_distortion = True
        else:
            self.has_distortion = False

    def compute_extrinsics(self, data: mujoco.MjData) -> Transform:
        """Compute ground-truth camera extrinsics (T_world_cam) in OpenCV coordinate conventions."""
        cam_pos_world = data.cam_xpos[self.cam_id].copy()
        cam_mat_mj = data.cam_xmat[self.cam_id].reshape(3, 3)  # Camera axes in world

        # Conversion matrix: X_cv = X_mj, Y_cv = -Y_mj, Z_cv = -Z_mj
        r_mj_to_cv = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

        # R_world_to_cam_cv: transforms point from world frame into OpenCV camera frame
        r_world_to_cv = r_mj_to_cv @ cam_mat_mj.T
        t_cv = -r_world_to_cv @ cam_pos_world

        # T_world_to_cam 4x4 matrix
        mat_world_to_cv = np.eye(4, dtype=np.float64)
        mat_world_to_cv[:3, :3] = r_world_to_cv
        mat_world_to_cv[:3, 3] = t_cv

        # Transform representing camera pose in world: T_world_cam = inv(T_world_to_cam)
        mat_cam_in_world = np.linalg.inv(mat_world_to_cv)
        t_world = tuple(float(x) for x in mat_cam_in_world[:3, 3])

        # Quaternion from 3x3 rotation
        from scipy.spatial.transform import Rotation

        q_xyzw = Rotation.from_matrix(mat_cam_in_world[:3, :3]).as_quat()
        q_tuple = (float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3]))

        return Transform(
            matrix=mat_cam_in_world.tolist(),
            translation=t_world,  # type: ignore[arg-type]
            rotation_xyzw=q_tuple,
        )

    def process_and_distort(
        self,
        raw_image_rgb: np.ndarray,
        sensor_noise_std: float = 1.0,
    ) -> np.ndarray:
        """Apply physical lens distortion and sensor noise to rendered image."""
        img = raw_image_rgb

        # Apply lens distortion
        if self.has_distortion:
            img = cv2.remap(img, self.mapx, self.mapy, interpolation=cv2.INTER_LINEAR)

        # Apply sensor noise (read noise + photon shot noise)
        if sensor_noise_std > 0.0:
            noise = self.rng.normal(0.0, sensor_noise_std, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        return img
