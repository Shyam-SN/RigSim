"""Rigid body 3D transformation representation."""

import numpy as np
from pydantic import BaseModel, Field, field_validator


class Transform(BaseModel):
    """Rigid body transform T_A_B (representing pose of B in frame A).

    Provides both 4x4 matrix and translation + unit quaternion [x, y, z, w].
    """

    matrix: list[list[float]] = Field(..., description="4x4 homogeneous transformation matrix")
    translation: tuple[float, float, float] = Field(..., description="x, y, z (meters)")
    rotation_xyzw: tuple[float, float, float, float] = Field(
        ..., description="Quaternion (x, y, z, w)"
    )

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 4 or any(len(row) != 4 for row in v):
            raise ValueError("Transform matrix must be 4x4")
        if (
            not np.isclose(v[3][0], 0.0)
            or not np.isclose(v[3][1], 0.0)
            or not np.isclose(v[3][2], 0.0)
            or not np.isclose(v[3][3], 1.0)
        ):
            raise ValueError("Bottom row of transform matrix must be [0, 0, 0, 1]")
        return v

    @classmethod
    def from_translation_quaternion(
        cls,
        translation: tuple[float, float, float],
        rotation_xyzw: tuple[float, float, float, float],
    ) -> "Transform":
        """Construct Transform from translation vector and xyzw quaternion."""
        x, y, z = translation
        qx, qy, qz, qw = rotation_xyzw
        norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-8:
            raise ValueError("Quaternion cannot have zero norm")
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

        # Compute 3x3 rotation matrix from normalized quaternion
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)

        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)

        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        mat = [
            [float(r00), float(r01), float(r02), float(x)],
            [float(r10), float(r11), float(r12), float(y)],
            [float(r20), float(r21), float(r22), float(z)],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return cls(
            matrix=mat,
            translation=(float(x), float(y), float(z)),
            rotation_xyzw=(float(qx), float(qy), float(qz), float(qw)),
        )

    @classmethod
    def from_matrix(cls, matrix: list[list[float]] | np.ndarray) -> "Transform":
        """Construct Transform from a 4x4 homogeneous transformation matrix."""
        from scipy.spatial.transform import Rotation

        mat = np.array(matrix, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"Matrix must have shape (4, 4), got {mat.shape}")

        R = mat[:3, :3]
        t = mat[:3, 3]

        # Extract unit quaternion [x, y, z, w]
        rot = Rotation.from_matrix(R)
        qx, qy, qz, qw = rot.as_quat()

        # Enforce exact bottom row [0, 0, 0, 1] for strict validation
        mat_clean = mat.copy()
        mat_clean[3, :] = [0.0, 0.0, 0.0, 1.0]

        return cls(
            matrix=mat_clean.tolist(),
            translation=(float(t[0]), float(t[1]), float(t[2])),
            rotation_xyzw=(float(qx), float(qy), float(qz), float(qw)),
        )

    @classmethod
    def identity(cls) -> "Transform":
        return cls(
            matrix=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            translation=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

    def as_numpy(self) -> np.ndarray:
        return np.array(self.matrix, dtype=np.float64)
