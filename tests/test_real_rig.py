"""Unit and integration tests for Milestone M7: Real Rig (Phone + Mac, QR clock sync, and labelling)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import qrcode
from typer.testing import CliRunner

from rigsim.cli import app
from rigsim.config.types import SensorKind
from rigsim.ingest.opencamera import OpenCameraSensorsAdapter
from rigsim.ingest.sensor_logger import SensorLoggerIngestAdapter
from rigsim.sync.qr_clock import QrClockSyncEstimator

runner = CliRunner()


@pytest.fixture
def mock_sensor_logger_export(tmp_path: Path) -> Path:
    """Create a synthetic Android Sensor Logger export folder with video and CSV IMU."""
    export_dir = tmp_path / "sensor_logger_raw"
    export_dir.mkdir(parents=True)

    # 1. Create a 15-frame video @ 30 FPS
    video_path = export_dir / "recording.mp4"
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))

    for k in range(15):
        frame = np.full((h, w, 3), int(100 + k * 8), dtype=np.uint8)
        cv2.putText(
            frame, f"Frame {k}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        out.write(frame)
    out.release()

    # 2. Create Accelerometer CSV
    base_ts = 1_700_000_000_000_000_000
    accel_lines = ["time,seconds_elapsed,x,y,z\n"]
    for i in range(50):
        t_ns = base_ts + i * 10_000_000  # 100 Hz
        t_s = i * 0.01
        accel_lines.append(f"{t_ns},{t_s:.3f},0.02,-0.03,9.81\n")

    with open(export_dir / "TotalAccelerometer.csv", "w") as f:
        f.writelines(accel_lines)

    # 3. Create Gyroscope CSV
    gyro_lines = ["time,seconds_elapsed,x,y,z\n"]
    for i in range(50):
        t_ns = base_ts + i * 10_000_000
        t_s = i * 0.01
        gyro_lines.append(f"{t_ns},{t_s:.3f},0.001,0.002,-0.001\n")

    with open(export_dir / "Gyroscope.csv", "w") as f:
        f.writelines(gyro_lines)

    return export_dir


@pytest.fixture
def mock_opencamera_export(tmp_path: Path) -> Path:
    """Create a synthetic OpenCamera Sensors folder."""
    export_dir = tmp_path / "opencamera_raw"
    export_dir.mkdir(parents=True)

    video_path = export_dir / "VID_20240101.mp4"
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))

    for k in range(15):
        frame = np.full((h, w, 3), int(120 + k * 5), dtype=np.uint8)
        out.write(frame)
    out.release()

    base_ts = 1_700_000_000_000_000_000
    imu_lines = ["time_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"]
    for i in range(50):
        t_ns = base_ts + i * 10_000_000
        imu_lines.append(f"{t_ns},0.01,0.02,9.80,0.001,-0.002,0.001\n")

    with open(export_dir / "imu.csv", "w") as f:
        f.writelines(imu_lines)

    return export_dir


def test_sensor_logger_ingest_adapter(mock_sensor_logger_export: Path, tmp_path: Path):
    """Test ingestion of Sensor Logger recordings into standard Session and MCAP."""
    out_dir = tmp_path / "ingested_phone"
    adapter = SensorLoggerIngestAdapter(
        input_path=mock_sensor_logger_export,
        output_dir=out_dir,
    )
    session = adapter.ingest()

    assert session.source == "phone"
    assert len(session.streams) == 2
    assert Path(session.mcap_path).exists()
    assert (out_dir / "session.json").exists()

    stream_map = {s.stream_id: s for s in session.streams}
    assert stream_map["phone_cam"].kind == SensorKind.RGB
    assert stream_map["phone_imu"].kind == SensorKind.IMU


def test_opencamera_sensors_adapter(mock_opencamera_export: Path, tmp_path: Path):
    """Test ingestion of OpenCamera Sensors recordings."""
    out_dir = tmp_path / "ingested_opencamera"
    adapter = OpenCameraSensorsAdapter(
        input_dir=mock_opencamera_export,
        output_dir=out_dir,
    )
    session = adapter.ingest()

    assert session.source == "phone"
    assert len(session.streams) == 2
    assert Path(session.mcap_path).exists()


def test_qr_clock_sync_estimator(tmp_path: Path):
    """Test optical QR clock decoding and robust Huber regression clock fitting."""
    estimator = QrClockSyncEstimator(display_refresh_rate_hz=60.0)

    # Simulate true physical relationship:
    # t_mac = a * t_phone + b
    # Target: drift = +20.0 PPM (slope a = 1.000020), initial offset b = +150.0 ms
    true_slope = 1.000020
    true_offset_ns = 150_000_000

    t0_phone = 1_700_000_000_000_000_000
    dt_phone_ns = 33_333_333  # ~30 FPS

    frames = []
    phone_ts_list = []

    for idx in range(20):
        p_ts = t0_phone + idx * dt_phone_ns
        # Calculate expected mac time with 1 ms simulated display quantization jitter
        m_ts = int(t0_phone + idx * dt_phone_ns * true_slope + true_offset_ns)

        # Generate actual QR image encoding the Mac timestamp
        payload = f'{{"t_ns": {m_ts}, "frame": {idx}}}'
        qr = qrcode.QRCode(border=2)
        qr.add_data(payload)
        qr.make(fit=True)
        mat = np.array(qr.get_matrix(), dtype=np.uint8)
        qr_img = np.where(mat, 0, 255).astype(np.uint8)
        qr_large = cv2.resize(qr_img, (350, 350), interpolation=cv2.INTER_NEAREST)

        frames.append(cv2.cvtColor(qr_large, cv2.COLOR_GRAY2BGR))
        phone_ts_list.append(p_ts)

    # Estimate clock sync
    res = estimator.process_frames(frames, phone_ts_list)

    assert res.decoded_count == 20
    assert res.decode_success_rate == 1.0
    # Drift recovery error < 5 PPM
    assert abs(res.drift_ppm - 20.0) < 5.0
    # Offset recovery error < 2.0 ms
    assert abs(res.offset_ms - 150.0) < 2.0
    # Residual jitter within bounds
    assert res.residual_jitter_std_ms < 5.0
    assert "display_quantization_ms" in res.error_budget_ms


def test_fault_labelling_cli(tmp_path: Path):
    """Test manual fault labelling CLI on a recording session."""
    sess_dir = tmp_path / "mock_session"
    sess_dir.mkdir(parents=True)
    manifest = sess_dir / "session.json"
    with open(manifest, "w") as f:
        json.dump(
            {
                "session_id": "test_sess",
                "source": "phone",
                "rig_config_name": "phone_rig",
                "created_at": "2026-01-01T00:00:00Z",
                "operator": "Tester",
                "streams": [],
                "mcap_path": str((sess_dir / "rec.mcap").resolve()),
            },
            f,
        )

    # Run label command via CliRunner
    result = runner.invoke(
        app,
        [
            "label",
            str(sess_dir),
            "--fault-type",
            "motion_blur",
            "--stream",
            "phone_cam",
            "--start",
            "1.5",
            "--end",
            "4.0",
            "--severity",
            "0.85",
        ],
    )
    assert result.exit_code == 0
    assert "Added label: motion_blur" in result.output

    labels_file = sess_dir / "labels.json"
    assert labels_file.exists()
    with open(labels_file) as f:
        labels = json.load(f)
    assert len(labels) == 1
    assert labels[0]["fault_type"] == "motion_blur"
    assert labels[0]["stream_id"] == "phone_cam"
    assert labels[0]["severity"] == 0.85
