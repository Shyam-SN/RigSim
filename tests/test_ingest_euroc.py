"""Unit and integration tests for EuRoC MAV ASL dataset ingestion, QA, and fault evaluation."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from rigsim.config.faults import FaultInjectionSpec, FaultScenarioConfig
from rigsim.config.types import FaultType, QAVerdict, SensorKind
from rigsim.eval.euroc_eval import EuRoCEvaluator
from rigsim.ingest.euroc import (
    EuRoCIngestAdapter,
    load_camera_sensor_yaml,
    load_imu_sensor_yaml,
)
from rigsim.qa.engine import QAEngine


@pytest.fixture
def mock_euroc_dataset(tmp_path: Path) -> Path:
    """Create a fully-compliant synthetic EuRoC MAV ASL directory structure."""
    euroc_root = tmp_path / "euroc_seq"
    mav0 = euroc_root / "mav0"
    cam0_dir = mav0 / "cam0"
    cam1_dir = mav0 / "cam1"
    imu0_dir = mav0 / "imu0"

    (cam0_dir / "data").mkdir(parents=True)
    (cam1_dir / "data").mkdir(parents=True)
    imu0_dir.mkdir(parents=True)

    # 1. Write cam0 sensor.yaml
    cam0_yaml = {
        "sensor_type": "camera",
        "comment": "VI-Sensor cam0 (MT9M034)",
        "T_BS": {
            "data": [
                0.01486,
                -0.99988,
                0.00414,
                -0.02164,
                0.99955,
                0.01496,
                0.02571,
                -0.06467,
                -0.02577,
                0.00375,
                0.99966,
                0.00981,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        },
        "rate_hz": 20.0,
        "resolution": [752, 480],
        "camera_model": "pinhole",
        "intrinsics": [458.654, 457.296, 367.215, 248.375],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [-0.2834, 0.0739, 0.00019, 0.000017],
    }
    with open(cam0_dir / "sensor.yaml", "w") as f:
        yaml.dump(cam0_yaml, f)

    # 2. Write cam1 sensor.yaml
    cam1_yaml = {
        "sensor_type": "camera",
        "comment": "VI-Sensor cam1 (MT9M034)",
        "T_BS": {
            "data": [
                0.01255,
                -0.99991,
                0.00441,
                0.08841,
                0.99958,
                0.01266,
                0.02558,
                -0.06461,
                -0.02563,
                0.00408,
                0.99966,
                0.00982,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        },
        "rate_hz": 20.0,
        "resolution": [752, 480],
        "camera_model": "pinhole",
        "intrinsics": [457.587, 456.134, 379.999, 255.238],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [-0.2836, 0.0745, 0.00018, 0.000019],
    }
    with open(cam1_dir / "sensor.yaml", "w") as f:
        yaml.dump(cam1_yaml, f)

    # 3. Write imu0 sensor.yaml
    imu0_yaml = {
        "sensor_type": "imu",
        "comment": "VI-Sensor imu0 (ADIS16448)",
        "T_BS": {
            "data": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        },
        "rate_hz": 200.0,
        "gyroscope_noise_density": 1.6968e-04,
        "gyroscope_random_walk": 1.9393e-05,
        "accelerometer_noise_density": 2.0000e-3,
        "accelerometer_random_walk": 3.0000e-3,
    }
    with open(imu0_dir / "sensor.yaml", "w") as f:
        yaml.dump(imu0_yaml, f)

    # 4. Generate 40 frames @ 20 Hz (2.0s duration) with realistic stationary initial phase
    # followed by dynamic visual-inertial rotation
    base_ts = 1_403_636_579_763_555_584
    num_frames = 40
    cam_dt_ns = 50_000_000  # 20 Hz (50 ms)

    cam0_csv_lines = ["#timestamp [ns],filename\n"]
    cam1_csv_lines = ["#timestamp [ns],filename\n"]

    w, h = 752, 480

    # Base textured pattern (natural gradient + dots to avoid scanline artifacts)
    np.random.seed(42)
    base_tex = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            base_tex[y, x] = int(120 + 30 * np.sin(x / 40.0) + 20 * np.cos(y / 35.0))

    # Add scattered circles and features
    for k in range(50):
        cx = int((k * 37) % (w - 100) + 50)
        cy = int((k * 29) % (h - 100) + 50)
        cv2.circle(base_tex, (cx, cy), 15 + (k % 12), int(50 + (k * 17) % 150), -1)

    def get_rotation_and_rate(t: float) -> tuple[float, float]:
        """Compute angle (deg) and angular velocity wz (rad/s). Starts stationary for 0.4s."""
        if t < 0.4:
            return 0.0, 0.0
        t_active = t - 0.4
        f1, f2 = 0.8, 1.6
        a1, a2 = 0.35, 0.15  # rad
        angle_rad = a1 * np.sin(2.0 * np.pi * f1 * t_active) + a2 * np.sin(
            2.0 * np.pi * f2 * t_active
        )
        wz_rad_s = a1 * 2.0 * np.pi * f1 * np.cos(
            2.0 * np.pi * f1 * t_active
        ) + a2 * 2.0 * np.pi * f2 * np.cos(2.0 * np.pi * f2 * t_active)
        return float(np.degrees(angle_rad)), float(wz_rad_s)

    center = (w // 2, h // 2)

    for i in range(num_frames):
        ts = base_ts + i * cam_dt_ns
        fn = f"{ts}.png"
        cam0_csv_lines.append(f"{ts},{fn}\n")
        cam1_csv_lines.append(f"{ts},{fn}\n")

        t_sec = i * 0.05
        angle_deg, _ = get_rotation_and_rate(t_sec)

        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        frame_cam0 = cv2.warpAffine(base_tex, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
        frame_cam1 = cv2.warpAffine(base_tex, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Add realistic sensor shot noise (sigma=1.0) so stationary frames have natural sensor variance
        noise0 = np.random.normal(0.0, 1.0, (h, w)).astype(np.float32)
        noise1 = np.random.normal(0.0, 1.0, (h, w)).astype(np.float32)
        frame_cam0 = np.clip(frame_cam0.astype(np.float32) + noise0, 0, 255).astype(np.uint8)
        frame_cam1 = np.clip(frame_cam1.astype(np.float32) + noise1, 0, 255).astype(np.uint8)

        cv2.imwrite(str(cam0_dir / "data" / fn), frame_cam0)
        cv2.imwrite(str(cam1_dir / "data" / fn), frame_cam1)

    with open(cam0_dir / "data.csv", "w") as f:
        f.writelines(cam0_csv_lines)
    with open(cam1_dir / "data.csv", "w") as f:
        f.writelines(cam1_csv_lines)

    # 5. Generate IMU samples @ 200 Hz
    imu_dt_ns = 5_000_000  # 200 Hz
    num_imu_samples = int(num_frames * (cam_dt_ns / imu_dt_ns))

    imu_csv_lines = [
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"
    ]

    for j in range(num_imu_samples):
        ts = base_ts + j * imu_dt_ns
        t_sec = j * 0.005
        _, wz = get_rotation_and_rate(t_sec)
        wx = 0.0005 * np.sin(t_sec)
        wy = 0.0005 * np.cos(t_sec)

        ax = 0.005 * np.sin(t_sec)
        ay = 0.005 * np.cos(t_sec)
        az = 9.80665

        imu_csv_lines.append(f"{ts},{wx:.6f},{wy:.6f},{wz:.6f},{ax:.6f},{ay:.6f},{az:.6f}\n")

    with open(imu0_dir / "data.csv", "w") as f:
        f.writelines(imu_csv_lines)

    return euroc_root


def test_parse_sensor_yamls(mock_euroc_dataset: Path):
    """Verify parsing of EuRoC camera and IMU sensor YAML configurations."""
    mav0 = mock_euroc_dataset / "mav0"
    cam0_cfg = load_camera_sensor_yaml(mav0 / "cam0" / "sensor.yaml")
    assert cam0_cfg.camera_model == "pinhole"
    assert cam0_cfg.resolution == (752, 480)
    assert cam0_cfg.rate_hz == 20.0
    assert abs(cam0_cfg.intrinsics.fx - 458.654) < 1e-3
    assert abs(cam0_cfg.distortion.coeffs[0] - (-0.2834)) < 1e-3

    imu_cfg = load_imu_sensor_yaml(mav0 / "imu0" / "sensor.yaml")
    assert imu_cfg.rate_hz == 200.0
    assert np.allclose(imu_cfg.T_BS.as_numpy(), np.eye(4))


def test_clean_euroc_ingest_and_qa_verdict(mock_euroc_dataset: Path, tmp_path: Path):
    """Test end-to-end ingestion of clean EuRoC dataset and verify QA engine verdict."""
    output_dir = tmp_path / "ingested_clean"
    adapter = EuRoCIngestAdapter(
        euroc_dir=mock_euroc_dataset,
        output_dir=output_dir,
        fault_config=None,
    )
    session = adapter.ingest()

    assert session.source == "euroc"
    assert len(session.streams) == 3
    assert Path(session.mcap_path).exists()
    assert (output_dir / "session.json").exists()

    # Verify stream kinds
    stream_map = {s.stream_id: s for s in session.streams}
    assert stream_map["cam0"].kind == SensorKind.STEREO_LEFT
    assert stream_map["cam1"].kind == SensorKind.STEREO_RIGHT
    assert stream_map["imu0"].kind == SensorKind.IMU

    # Execute QA Engine
    engine = QAEngine()
    verdict = engine.process_session(output_dir)

    assert verdict.overall_verdict in (QAVerdict.PASS, QAVerdict.WARN)
    assert verdict.usable_percentage >= 80.0

    # Timing checks T1 (drops), T2 (FPS), T3 (jitter) should pass cleanly
    for res in verdict.results:
        if res.check_id in (
            "T1_FRAME_DROPS",
            "T2_EFFECTIVE_FPS",
            "T3_FPS_JITTER",
            "T8_IMU_CONTINUITY",
        ):
            assert res.verdict == QAVerdict.PASS, (
                f"Check {res.check_id} failed on clean data: {res.explanation}"
            )


def test_euroc_with_injected_faults_on_real_images(mock_euroc_dataset: Path, tmp_path: Path):
    """Inject synthetic faults (drops, blur, underexposure) onto real images and verify QA catches them."""
    output_dir = tmp_path / "ingested_faults"

    # Define fault scenario on cam0 and imu0
    fault_cfg = FaultScenarioConfig(
        scenario_name="euroc_test_faults",
        description="EuRoC test faults",
        faults=[
            FaultInjectionSpec(
                fault_type=FaultType.BURST_FRAME_DROP,
                stream_id="cam0",
                t_start_s=0.3,
                t_end_s=0.6,
                severity=1.0,
            ),
            FaultInjectionSpec(
                fault_type=FaultType.MOTION_BLUR,
                stream_id="cam1",
                t_start_s=0.2,
                t_end_s=0.8,
                severity=0.8,
            ),
            FaultInjectionSpec(
                fault_type=FaultType.UNDEREXPOSURE,
                stream_id="cam0",
                t_start_s=0.5,
                t_end_s=0.9,
                severity=0.7,
            ),
        ],
    )

    adapter = EuRoCIngestAdapter(
        euroc_dir=mock_euroc_dataset,
        output_dir=output_dir,
        fault_config=fault_cfg,
    )
    session = adapter.ingest()

    assert session.ground_truth_labels_path is not None
    assert Path(session.ground_truth_labels_path).exists()

    evaluator = EuRoCEvaluator()
    summary = evaluator.run_fault_evaluation(
        euroc_dir=mock_euroc_dataset,
        output_dir=output_dir,
        fault_config=fault_cfg,
    )

    assert summary["total_faults"] == 3
    assert summary["detected_faults"] >= 2
    assert str(summary["verdict"]).upper() in ("WARN", "FAIL")


def test_euroc_imu_cam_offset_sweep(mock_euroc_dataset: Path, tmp_path: Path):
    """Test IMU-camera temporal calibration recovery across injected shifts on EuRoC."""
    evaluator = EuRoCEvaluator()
    sweep_tmp = tmp_path / "sweep_run"

    test_offsets = [-15.0, 0.0, 15.0]
    results = evaluator.run_offset_sweep(
        euroc_dir=mock_euroc_dataset,
        tmp_base=sweep_tmp,
        offsets_ms=test_offsets,
        max_frames=40,
    )

    assert len(results) == len(test_offsets)
    for res in results:
        # Cross-correlation peak should be positive
        assert res.correlation_peak > 0.0
        # Recovery error should be well bounded
        assert res.estimation_error_ms < 5.0
