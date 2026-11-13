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
from rigsim.qa.timing import (
    check_t1_frame_drops,
    check_t2_effective_fps,
    check_t3_jitter,
    check_t4_timestamp_sanity,
    check_t5_transport_latency,
    check_t6_inter_camera_sync,
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
    "check_t8_imu_continuity",
    "check_i1_blur",
    "check_i2_exposure",
    "check_i3_occlusion",
    "check_i4_frozen_frames",
    "check_i5_corruption",
    "check_i6_noise_level",
]
