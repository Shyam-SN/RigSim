"""Enumerations and foundational types for RigSim."""

from enum import StrEnum


class SensorKind(StrEnum):
    RGB = "rgb"
    STEREO_LEFT = "stereo_left"
    STEREO_RIGHT = "stereo_right"
    IMU = "imu"


class DistortionModel(StrEnum):
    RADTAN = "radtan"  # OpenCV standard: [k1, k2, p1, p2, (k3)]
    EQUIDISTANT = "equidistant"  # Fisheye / Kannala-Brandt
    NONE = "none"


class SyncMode(StrEnum):
    HARDWARE_TRIGGER = "hardware_trigger"
    PTP_SOFTWARE = "ptp_software"
    INDEPENDENT = "independent"


class SessionSource(StrEnum):
    SIM = "sim"
    EUROC = "euroc"
    PHONE = "phone"
    WEBCAM = "webcam"
    PHONE_WEBCAM = "phone+webcam"


class FaultType(StrEnum):
    # Timing / Transport faults (1-12)
    RANDOM_FRAME_DROP = "random_frame_drop"
    BURST_FRAME_DROP = "burst_frame_drop"
    FPS_JITTER = "fps_jitter"
    LOW_EFFECTIVE_FPS = "low_effective_fps"
    CLOCK_DRIFT = "clock_drift"
    CLOCK_OFFSET = "clock_offset"
    IMU_CAM_OFFSET = "imu_cam_offset"
    CLOCK_STEP = "clock_step"
    TIMESTAMP_NON_MONOTONIC = "timestamp_non_monotonic"
    TIMESTAMP_DUPLICATE = "timestamp_duplicate"
    HOST_LATENCY_SPIKE = "host_latency_spike"
    IMU_DROPOUT = "imu_dropout"
    STREAM_DISCONNECT = "stream_disconnect"

    # Image Content faults (13-20)
    MOTION_BLUR = "motion_blur"
    OVEREXPOSURE = "overexposure"
    UNDEREXPOSURE = "underexposure"
    LENS_OCCLUSION = "lens_occlusion"
    FROZEN_FRAME = "frozen_frame"
    CORRUPTED_FRAME = "corrupted_frame"
    HEAVY_NOISE = "heavy_noise"
    CAMERA_BUMP = "camera_bump"
    INTRINSIC_SHIFT = "intrinsic_shift"

    # IMU Content faults (21-24)
    IMU_SATURATION = "imu_saturation"
    IMU_BIAS_JUMP = "imu_bias_jump"
    IMU_HIGH_NOISE = "imu_high_noise"
    IMU_AXIS_FLIP = "imu_axis_flip"


class QAVerdict(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
