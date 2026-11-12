"""Unit tests for simulation, kinematics, optics, IMU, and MCAP serialization."""

from pathlib import Path

import cv2
import mujoco
import numpy as np
from mcap.reader import make_reader

from rigsim.clocks.model import ClockEngine
from rigsim.config.clock import ClockModel, ClockStepEvent
from rigsim.config.imu import IMUSpec
from rigsim.config.rig import RigConfig
from rigsim.sim.camera import CameraSimulator
from rigsim.sim.imu import IMUSimulator
from rigsim.sim.simulator import SimulationEngine


def test_ground_truth_intrinsics_and_opencv_projection():
    """Verify ground truth camera intrinsics and extrinsics match OpenCV conventions."""
    model = mujoco.MjModel.from_xml_path("assets/mjcf/tabletop_6cam.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Use tabletop rig config
    import yaml

    with open("configs/rigs/tabletop_6cam.yaml") as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    cam_spec = rig_cfg.get_camera("cam_stereo_left")
    assert cam_spec is not None

    cam_sim = CameraSimulator(cam_spec, model, seed=42)
    extrinsics = cam_sim.compute_extrinsics(data)

    # OpenCV projection matrices
    T_world_cam = extrinsics.as_numpy()
    T_cam_world = np.linalg.inv(T_world_cam)
    R_cv = T_cam_world[:3, :3]
    t_cv = T_cam_world[:3, 3]

    # Target point: center of red_cube in world
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_cube")
    p_world = data.geom_xpos[geom_id].copy()

    # OpenCV cv2.projectPoints
    rvec, _ = cv2.Rodrigues(R_cv)
    points_3d = np.array([p_world], dtype=np.float64)
    dist_zero = np.zeros(4, dtype=np.float64)
    proj_cv2, _ = cv2.projectPoints(points_3d, rvec, t_cv, cam_sim.K, dist_zero)
    u_cv2, v_cv2 = proj_cv2[0, 0]

    # Analytic projection
    p_cam = R_cv @ p_world + t_cv
    assert p_cam[2] > 0, "Point must be in front of camera optical axis"
    u_analytic = cam_sim.intrinsics.fx * (p_cam[0] / p_cam[2]) + cam_sim.intrinsics.cx
    v_analytic = cam_sim.intrinsics.fy * (p_cam[1] / p_cam[2]) + cam_sim.intrinsics.cy

    # Exact numeric equivalence between cv2 and analytic projection
    np.testing.assert_allclose(u_cv2, u_analytic, atol=1e-5)
    np.testing.assert_allclose(v_cv2, v_analytic, atol=1e-5)

    # Render image and verify that pixel at (u, v) is distinctly red
    renderer = mujoco.Renderer(model, cam_sim.height, cam_sim.width)
    renderer.update_scene(data, camera="cam_stereo_left")
    rendered_rgb = renderer.render()
    renderer.close()

    pixel_rgb = rendered_rgb[int(round(v_cv2)), int(round(u_cv2))]
    # Red cube: high red channel compared to green/blue
    assert pixel_rgb[0] > 180
    assert pixel_rgb[0] > pixel_rgb[1] + 40


def test_imu_simulation_noise_and_gravity():
    """Verify IMU discrete noise statistics and static gravity norm."""
    spec = IMUSpec(
        stream_id="imu_test",
        nominal_rate_hz=200.0,
        clock_id="clock_imu",
        frame_id="imu_link",
        gyro_noise_density=1.6968e-04,
        gyro_random_walk=1.9393e-05,
        accel_noise_density=2.0e-03,
        accel_random_walk=3.0e-03,
        extrinsics_to_rig={"translation": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
    )
    imu = IMUSimulator(spec, seed=42)

    # Simulate 2 seconds of stationary sensor on a table (pure 1g gravity in +z)
    accel_samples = []
    gyro_samples = []
    true_accel = np.array([0.0, 0.0, 9.81])
    true_gyro = np.array([0.0, 0.0, 0.0])
    true_quat = np.array([1.0, 0.0, 0.0, 0.0])

    for i in range(400):  # 400 samples at 200 Hz = 2.0 s
        t_sim_s = i * 0.005
        sample = imu.update_and_sample(
            true_accel=true_accel,
            true_gyro=true_gyro,
            true_quat_wxyz=true_quat,
            device_time_ns=int(t_sim_s * 1e9),
            host_time_ns=int(t_sim_s * 1e9 + 5e6),
            t_sim_s=t_sim_s,
        )
        accel_samples.append(sample.linear_acceleration)
        gyro_samples.append(sample.angular_velocity)

    accel_arr = np.array(accel_samples)
    gyro_arr = np.array(gyro_samples)

    # Mean acceleration norm should be close to 9.81 m/s^2 (within turn-on bias)
    mean_accel_norm = np.mean(np.linalg.norm(accel_arr, axis=1))
    assert 9.6 < mean_accel_norm < 10.0

    # Mean gyro angular speed should be near 0 (< 0.02 rad/s turn-on bias)
    mean_gyro_norm = np.mean(np.linalg.norm(gyro_arr, axis=1))
    assert mean_gyro_norm < 0.05


def test_clock_model_drift_and_jitter():
    """Verify clock engine frequency drift and step events."""
    clock_model = ClockModel(
        clock_id="test_clock",
        offset_ns=1_000_000,  # 1 ms offset
        drift_ppm=100.0,  # 100 ppm = 100 ns per 1 ms
        jitter_std_ns=1000.0,  # 1 microsecond jitter
        step_events=[
            ClockStepEvent(t_sim_ns=5_000_000_000, step_ns=50_000_000)
        ],  # 50 ms step at 5s
    )
    engine = ClockEngine(clock_model, seed=123)

    # At t=0
    t0_dev = engine.sim_to_device_time_ns(0)
    # Approx 1 ms offset +- jitter
    assert np.isclose(t0_dev, 1_000_000, atol=10_000)

    # At t=1s (1e9 ns) without step event:
    # Expected: (1 + 100e-6)*1e9 + 1e6 = 1_000_100_000 + 1_000_000 = 1_001_100_000 ns
    t1_dev = engine.sim_to_device_time_ns(1_000_000_000)
    assert np.isclose(t1_dev, 1_001_100_000, atol=10_000)

    # At t=6s (after 5s step event of 50 ms):
    t6_dev = engine.sim_to_device_time_ns(6_000_000_000)
    expected_t6 = int(round((1.0 + 100e-6) * 6e9 + 1e6 + 50e6))
    assert np.isclose(t6_dev, expected_t6, atol=15_000)


def test_simulation_run_and_mcap_inspection(tmp_path: Path):
    """Run full simulation on handheld rig and verify MCAP integrity."""
    import yaml

    with open("configs/rigs/handheld_stereo_imu.yaml") as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    engine = SimulationEngine(
        rig_config=rig_cfg,
        duration_s=1.0,  # 1 second quick test
        seed=42,
        output_dir=tmp_path,
    )
    session = engine.run()

    assert session.session_id.startswith("sim_handheld_stereo_imu")
    mcap_file = Path(session.mcap_path)
    assert mcap_file.exists()
    assert mcap_file.stat().st_size > 500_000  # > 500 KB

    # Read back with mcap reader
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        assert summary is not None

        # Verify channel topics
        topics = {ch.topic for ch in summary.channels.values()}
        assert "/cam0/image_compressed" in topics
        assert "/cam1/image_compressed" in topics
        assert "/cam0/calibration" in topics
        assert "/cam1/calibration" in topics
        assert "/imu0/data" in topics
        assert "/tf" in topics

        # Verify attachments
        attachments = [a.name for a in reader.iter_attachments()]
        assert "rig.yaml" in attachments
