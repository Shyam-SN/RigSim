"""Unit tests for all 24 fault injectors and ground truth label matching."""

import cv2
import numpy as np

from rigsim.config.camera import CameraIntrinsics
from rigsim.config.transform import Transform
from rigsim.config.types import FaultType
from rigsim.faults.engine import FaultEngine
from rigsim.faults.image import (
    CameraBumpInjector,
    CorruptedFrameInjector,
    ExposureFaultInjector,
    FrozenFrameInjector,
    HeavyNoiseInjector,
    IntrinsicShiftInjector,
    LensOcclusionInjector,
    MotionBlurInjector,
)
from rigsim.faults.imu import (
    IMUAxisFlipInjector,
    IMUBiasJumpInjector,
    IMUHighNoiseInjector,
    IMUSaturationInjector,
)
from rigsim.faults.timing import (
    BurstFrameDropInjector,
    ClockDriftInjector,
    ClockOffsetInjector,
    ClockStepInjector,
    FPSJitterInjector,
    HostLatencySpikeInjector,
    IMUCamOffsetInjector,
    IMUDropoutInjector,
    LowEffectiveFPSInjector,
    RandomFrameDropInjector,
    StreamDisconnectInjector,
    TimestampSanityInjector,
)
from rigsim.sim.imu import IMUSample

# --- TIMING FAULTS (1-12) ---


def test_fault_1_random_frame_drop():
    inj = RandomFrameDropInjector("cam0", 100, 200, severity=1.0, drop_probability=1.0)
    assert not inj.should_drop(50)
    assert inj.should_drop(150)
    assert not inj.should_drop(250)
    label = inj.create_label()
    assert label.fault_type == FaultType.RANDOM_FRAME_DROP
    assert label.t_start_ns == 100


def test_fault_2_burst_frame_drop():
    inj = BurstFrameDropInjector("cam0", 100, 200, burst_size=3)
    assert not inj.should_drop(50)
    assert inj.should_drop(110)
    assert inj.should_drop(120)
    assert inj.should_drop(130)
    assert not inj.should_drop(140)  # Exhausted burst size
    label = inj.create_label()
    assert label.fault_type == FaultType.BURST_FRAME_DROP
    assert label.params["burst_size"] == 3


def test_fault_3_fps_jitter():
    inj = FPSJitterInjector(
        "cam0",
        t_start_ns=100_000_000,
        t_end_ns=200_000_000,
        severity=1.0,
        jitter_std_ms=10.0,
        seed=42,
    )
    t_in = 150_000_000
    t_out = inj.modify_timestamp(t_in)
    assert t_out != t_in
    assert abs(t_out - t_in) < 100_000_000
    assert inj.modify_timestamp(50_000_000) == 50_000_000  # Inactive
    label = inj.create_label()
    assert label.fault_type == FaultType.FPS_JITTER


def test_fault_4_low_effective_fps():
    inj = LowEffectiveFPSInjector("cam0", 100, 200, target_fps_ratio=0.5)
    # Throttling by 50% should alternate drops
    drops = [inj.should_drop(150) for _ in range(4)]
    assert drops == [True, False, True, False]
    label = inj.create_label()
    assert label.fault_type == FaultType.LOW_EFFECTIVE_FPS


def test_fault_5_clock_drift():
    inj = ClockDriftInjector("cam0", 1_000_000_000, 10_000_000_000, drift_ppm=100.0)
    # 1 second after start: delta = 1e9 * 100e-6 = 100_000 ns
    t_mod = inj.modify_timestamp(2_000_000_000)
    assert t_mod == 2_000_100_000
    label = inj.create_label()
    assert label.fault_type == FaultType.CLOCK_DRIFT


def test_fault_6_clock_offset():
    inj = ClockOffsetInjector("cam0", 100, 200, offset_ms=15.0)
    assert inj.modify_timestamp(50) == 50
    assert inj.modify_timestamp(150) == 150 + 15_000_000
    label = inj.create_label()
    assert label.fault_type == FaultType.CLOCK_OFFSET


def test_fault_7_imu_cam_offset():
    inj = IMUCamOffsetInjector("imu0", 100, 200, offset_ms=25.0)
    assert inj.modify_timestamp(150) == 150 + 25_000_000
    label = inj.create_label()
    assert label.fault_type == FaultType.IMU_CAM_OFFSET


def test_fault_8_clock_step():
    inj = ClockStepInjector("cam0", 5_000_000_000, 10_000_000_000, step_ms=50.0)
    assert inj.modify_timestamp(4_000_000_000) == 4_000_000_000
    assert inj.modify_timestamp(5_000_000_000) == 5_050_000_000
    label = inj.create_label()
    assert label.fault_type == FaultType.CLOCK_STEP


def test_fault_9_timestamp_sanity():
    inj_dup = TimestampSanityInjector("cam0", 100, 200, mode="duplicate")
    inj_dup.modify_timestamp(120)
    dup_ts = inj_dup.modify_timestamp(140)
    assert dup_ts == 120

    inj_nonmono = TimestampSanityInjector("cam0", 100, 200, mode="non_monotonic")
    inj_nonmono.modify_timestamp(120)
    nonmono_ts = inj_nonmono.modify_timestamp(140)
    assert nonmono_ts < 120
    label = inj_nonmono.create_label()
    assert label.fault_type == FaultType.TIMESTAMP_NON_MONOTONIC


def test_fault_10_host_latency_spike():
    inj = HostLatencySpikeInjector("cam0", 100, 200, spike_delay_ms=60.0)
    assert inj.modify_host_time(155_000_000, 150) == 155_000_000 + 60_000_000
    assert inj.modify_host_time(55_000_000, 50) == 55_000_000
    label = inj.create_label()
    assert label.fault_type == FaultType.HOST_LATENCY_SPIKE


def test_fault_11_imu_dropout():
    inj = IMUDropoutInjector("imu0", 100, 200)
    assert not inj.should_drop(50)
    assert inj.should_drop(150)
    label = inj.create_label()
    assert label.fault_type == FaultType.IMU_DROPOUT


def test_fault_12_stream_disconnect():
    inj = StreamDisconnectInjector("cam0", 100, 200)
    assert not inj.should_drop(50)
    assert inj.should_drop(100)
    assert inj.should_drop(300)  # Stays dead
    label = inj.create_label()
    assert label.fault_type == FaultType.STREAM_DISCONNECT


# --- IMAGE CONTENT FAULTS (13-20) ---


def test_fault_13_motion_blur():
    inj = MotionBlurInjector("cam0", 100, 200, severity=1.0, kernel_size=15)
    # Checkerboard image with high edge variance
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[::10, :] = 255
    var_sharp = cv2.Laplacian(img, cv2.CV_64F).var()

    blurred = inj.modify_image(img, 150)
    var_blurred = cv2.Laplacian(blurred, cv2.CV_64F).var()

    assert var_blurred < var_sharp * 0.7  # Significant blur
    label = inj.create_label()
    assert label.fault_type == FaultType.MOTION_BLUR


def test_fault_14_exposure_faults():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)

    inj_over = ExposureFaultInjector("cam0", 100, 200, mode="overexposure", gain=3.0)
    over = inj_over.modify_image(img, 150)
    assert over.mean() > 240

    inj_under = ExposureFaultInjector("cam0", 100, 200, mode="underexposure", gain=0.1)
    under = inj_under.modify_image(img, 150)
    assert under.mean() < 30


def test_fault_15_lens_occlusion():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    inj = LensOcclusionInjector("cam0", 100, 200, mode="full_black")
    occluded = inj.modify_image(img, 150)
    assert np.all(occluded == 0)

    inj_finger = LensOcclusionInjector("cam0", 100, 200, mode="finger_blob")
    blob = inj_finger.modify_image(img, 150)
    assert blob.mean() < img.mean()


def test_fault_16_frozen_frame():
    inj = FrozenFrameInjector("cam0", 100, 200)
    frame1 = np.full((20, 20, 3), 50, dtype=np.uint8)
    frame2 = np.full((20, 20, 3), 100, dtype=np.uint8)

    out1 = inj.modify_image(frame1, 120)
    out2 = inj.modify_image(frame2, 140)  # Should remain frame1
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(out2, frame1)


def test_fault_17_corrupted_frame():
    inj = CorruptedFrameInjector("cam0", 100, 200)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    corrupted = inj.modify_image(img, 150)
    assert np.any(corrupted > 0)


def test_fault_18_heavy_noise():
    inj = HeavyNoiseInjector("cam0", 100, 200, noise_std=50.0)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    noisy = inj.modify_image(img, 150)
    assert noisy.std() > 30.0


def test_fault_19_camera_bump():
    inj = CameraBumpInjector("cam0", 100, 200, delta_rotation_deg=5.0, delta_translation_mm=10.0)
    t_orig = Transform.identity()
    t_bumped = inj.modify_extrinsics(t_orig, 150)
    assert not np.allclose(t_orig.as_numpy(), t_bumped.as_numpy())
    assert abs(t_bumped.translation[0] - t_orig.translation[0]) > 0.005


def test_fault_20_intrinsic_shift():
    inj = IntrinsicShiftInjector("cam0", 100, 200, focal_shift_ratio=0.10)
    k_orig = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    k_shifted = inj.modify_intrinsics(k_orig, 150)
    assert np.isclose(k_shifted.fx, 550.0)
    assert np.isclose(k_shifted.fy, 550.0)


# --- IMU CONTENT FAULTS (21-24) ---


def _dummy_imu_sample(ts: int = 150) -> IMUSample:
    return IMUSample(
        device_time_ns=ts,
        host_time_ns=ts + 5_000_000,
        linear_acceleration=np.array([0.1, 0.2, 9.81]),
        angular_velocity=np.array([0.01, -0.02, 0.03]),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        temperature_c=25.0,
        is_saturated=False,
    )


def test_fault_21_imu_saturation():
    inj = IMUSaturationInjector("imu0", 100, 200, axis="gyro_z")
    s = _dummy_imu_sample(150)
    mod = inj.modify_sample(s)
    assert mod.is_saturated
    assert np.isclose(mod.angular_velocity[2], 34.906)


def test_fault_22_imu_bias_jump():
    inj = IMUBiasJumpInjector("imu0", 100, 200, accel_jump_m_s2=2.0, gyro_jump_rad_s=0.5)
    s = _dummy_imu_sample(150)
    mod = inj.modify_sample(s)
    assert np.isclose(mod.linear_acceleration[2] - s.linear_acceleration[2], 2.0)
    assert np.isclose(mod.angular_velocity[2] - s.angular_velocity[2], 0.5)


def test_fault_23_imu_high_noise():
    inj = IMUHighNoiseInjector("imu0", 100, 200, noise_multiplier=10.0, seed=42)
    s = _dummy_imu_sample(150)
    mod = inj.modify_sample(s)
    assert not np.allclose(s.linear_acceleration, mod.linear_acceleration)


def test_fault_24_imu_axis_flip():
    inj = IMUAxisFlipInjector("imu0", 100, 200, flip_axis="z")
    s = _dummy_imu_sample(150)
    mod = inj.modify_sample(s)
    assert np.isclose(mod.linear_acceleration[2], -s.linear_acceleration[2])
    assert np.isclose(mod.angular_velocity[2], -s.angular_velocity[2])


# --- FAULT ENGINE INTEGRATION TEST ---


def test_fault_engine_scenario_configuration():
    import yaml

    from rigsim.config.faults import FaultScenarioConfig

    with open("configs/faults/realistic_mixed.yaml") as f:
        cfg = FaultScenarioConfig.model_validate(yaml.safe_load(f))

    engine = FaultEngine(cfg, seed=42)
    assert len(engine.injectors) == len(cfg.faults)
    labels = engine.get_ground_truth_labels()
    assert len(labels) == len(cfg.faults)
    # Check that each label accurately reflects its injection spec
    types_in_labels = {lbl.fault_type for lbl in labels}
    assert FaultType.RANDOM_FRAME_DROP in types_in_labels
    assert FaultType.CAMERA_BUMP in types_in_labels
    assert FaultType.IMU_SATURATION in types_in_labels
