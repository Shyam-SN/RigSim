"""RigSim Automated QA Engine module."""

from rigsim.qa.engine import QAEngine
from rigsim.qa.image import (
    check_i1_blur,
    check_i2_exposure,
    check_i3_occlusion,
    check_i4_frozen_frames,
    check_i5_corruption,
    check_i6_noise_level,
)
from rigsim.qa.imu import (
    check_m1_imu_gravity,
    check_m2_imu_static_bias,
    check_m3_imu_bias_jump,
    check_m4_imu_gravity_direction,
)
from rigsim.qa.timing import (
    check_t1_frame_drops,
    check_t2_effective_fps,
    check_t3_jitter,
    check_t4_timestamp_sanity,
    check_t5_transport_latency,
    check_t6_inter_camera_sync,
    check_t7_imu_cam_offset,
    check_t8_imu_continuity,
)
from rigsim.qa.verdict import VerdictEvaluator

__all__ = [
    "QAEngine",
    "VerdictEvaluator",
    "check_t1_frame_drops",
    "check_t2_effective_fps",
    "check_t3_jitter",
    "check_t4_timestamp_sanity",
    "check_t5_transport_latency",
    "check_t6_inter_camera_sync",
    "check_t7_imu_cam_offset",
    "check_t8_imu_continuity",
    "check_i1_blur",
    "check_i2_exposure",
    "check_i3_occlusion",
    "check_i4_frozen_frames",
    "check_i5_corruption",
    "check_i6_noise_level",
    "check_m1_imu_gravity",
    "check_m2_imu_static_bias",
    "check_m3_imu_bias_jump",
    "check_m4_imu_gravity_direction",
]

