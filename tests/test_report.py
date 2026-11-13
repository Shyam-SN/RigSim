"""Unit tests for Milestone M5: Single-session report, batch report, and Foxglove MCAP annotations."""

import json
from pathlib import Path

import numpy as np
import pytest
from mcap.reader import make_reader

from rigsim.config.transform import Transform
from rigsim.config.camera import CameraIntrinsics
from rigsim.config.qa import QAResult, QAVerdict, SessionVerdict, TimeWindow
from rigsim.config.session import Session, SessionSource, Stream
from rigsim.config.types import SensorKind
from rigsim.io.mcap_writer import RigSimMCAPWriter
from rigsim.report.batch_report import generate_batch_report
from rigsim.report.foxglove import export_foxglove_annotations
from rigsim.report.session_report import generate_session_report
from rigsim.sim.imu import IMUSample


@pytest.fixture
def mock_session_dir(tmp_path: Path) -> Path:
    """Create a minimal valid session folder with MCAP, session.json and qa/results.json."""
    sess_dir = tmp_path / "test_session_01"
    sess_dir.mkdir(parents=True, exist_ok=True)
    mcap_path = sess_dir / "session.mcap"

    # Write small valid MCAP
    writer = RigSimMCAPWriter(mcap_path)
    writer.register_camera_channel("cam_wrist")
    writer.register_imu_channel("imu_rig")
    img = np.full((120, 160, 3), 128, dtype=np.uint8)

    # 5 frames
    for i in range(5):
        t_ns = 1_700_000_000_000_000_000 + i * 33_333_333
        writer.write_image("cam_wrist", "cam_wrist_optical", img, t_ns, t_ns + 5_000_000)
        imu_s = IMUSample(
            device_time_ns=t_ns,
            host_time_ns=t_ns + 2_000_000,
            linear_acceleration=np.array([0.0, 0.0, 9.81]),
            angular_velocity=np.array([0.01, 0.02, 0.03]),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            temperature_c=24.5,
        )
        writer.write_imu("imu_rig", imu_s)
    writer.close()

    # Write session.json
    session = Session(
        session_id="test_session_01",
        source=SessionSource.SIM,
        rig_config_name="tabletop_6cam",
        created_at="2023-11-13T12:00:00Z",
        operator="qa_lead",
        mcap_path=str(mcap_path),
        streams=[
            Stream(
                stream_id="cam_wrist",
                kind=SensorKind.RGB,
                nominal_rate_hz=30.0,
                resolution=[160, 120],
                clock_id="clock_wrist",
                frame_id="cam_wrist_optical",
                extrinsics=Transform.identity(),
                intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=80.0, cy=60.0, width=160, height=120, fov_deg=60.0),
            ),
            Stream(
                stream_id="imu_rig",
                kind=SensorKind.IMU,
                nominal_rate_hz=200.0,
                clock_id="clock_imu",
                frame_id="imu_rig_link",
                extrinsics=Transform.identity(),
            ),
        ],
    )
    with open(sess_dir / "session.json", "w") as f:
        f.write(session.model_dump_json(indent=2))

    # Write qa/results.json
    qa_dir = sess_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    qa_results = [
        QAResult(
            check_id="T1_FRAME_DROPS",
            stream_id="cam_wrist",
            verdict=QAVerdict.PASS,
            metric_name="drop_ratio",
            value=0.0,
            threshold=0.01,
            explanation="Zero frame drops.",
        ),
        QAResult(
            check_id="I3_OCCLUSION",
            stream_id="cam_wrist",
            verdict=QAVerdict.FAIL,
            metric_name="occluded_frames_count",
            value=1.0,
            threshold=0.0,
            flagged_windows=[
                TimeWindow(
                    t_start_ns=1_700_000_000_000_000_000 + 33_333_333,
                    t_end_ns=1_700_000_000_000_000_000 + 66_666_666,
                )
            ],
            explanation="FAIL: Lens occlusion detected on frame 1.",
        ),
    ]

    verdict = SessionVerdict(
        session_id="test_session_01",
        overall_verdict=QAVerdict.FAIL,
        usable_time_ranges=[
            TimeWindow(
                t_start_ns=1_700_000_000_000_000_000,
                t_end_ns=1_700_000_000_000_000_000 + 33_333_333,
            )
        ],
        usable_duration_s=0.033,
        total_duration_s=0.166,
        yield_ratio=0.20,
        summary_rules=["Lens occlusion detected on cam_wrist."],
        results=qa_results,
    )

    with open(qa_dir / "results.json", "w") as f:
        f.write(verdict.model_dump_json(indent=2))

    return sess_dir


def test_foxglove_annotation_export(mock_session_dir: Path) -> None:
    """Verify that export_foxglove_annotations adds /qa/events and /qa/logs to MCAP."""
    mcap_path = mock_session_dir / "session.mcap"
    with open(mock_session_dir / "qa" / "results.json") as f:
        verdict = SessionVerdict.model_validate_json(f.read())

    out_mcap = export_foxglove_annotations(
        mcap_path=mcap_path,
        qa_results=verdict.results,
        session_verdict=verdict,
    )
    assert out_mcap.exists()

    # Read back and inspect channels
    with open(out_mcap, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        topics = [c.topic for c in summary.channels.values()]

        assert "/qa/events" in topics
        assert "/qa/logs" in topics

        events_count = 0
        logs_count = 0
        for _, channel, _ in reader.iter_messages():
            if channel.topic == "/qa/events":
                events_count += 1
            elif channel.topic == "/qa/logs":
                logs_count += 1

        assert events_count >= 1
        assert logs_count >= 2  # defect log + summary log


def test_single_session_report_generation(mock_session_dir: Path) -> None:
    """Verify that generate_session_report creates offline self-contained HTML."""
    out_html = generate_session_report(mock_session_dir)
    assert out_html.exists()
    assert out_html.name == "report.html"

    content = out_html.read_text(encoding="utf-8")
    assert "RigSim QA Analysis Report" in content
    assert "test_session_01" in content
    assert "badge-FAIL" in content
    assert "Multi-Stream Temporal Defect Timeline" in content
    assert "I3_OCCLUSION" in content
    assert "Lens occlusion detected on frame 1" in content
    # Offline self-contained Plotly JS verification
    assert "Plotly.newPlot" in content or "plotly" in content.lower()


def test_batch_summary_report_generation(tmp_path: Path, mock_session_dir: Path) -> None:
    """Verify that generate_batch_report creates fleet summary HTML."""
    out_html = generate_batch_report(tmp_path)
    assert out_html.exists()
    assert out_html.name == "summary_report.html"

    content = out_html.read_text(encoding="utf-8")
    assert "RigSim Fleet QA Operations Summary" in content
    assert "Fleet Session Verdicts" in content
    assert "test_session_01" in content
    assert "Top QA Failure Root Causes" in content
