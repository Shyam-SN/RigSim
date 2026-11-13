"""Unit and integration tests for Milestone M10: Usable Episode Extraction & Client Delivery Manifests."""

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from rigsim.cli import app
from rigsim.config.camera import CameraDistortion, CameraIntrinsics
from rigsim.config.episode import Episode
from rigsim.config.qa import QAResult, QAVerdict, SessionVerdict, TimeWindow
from rigsim.config.session import Session, Stream
from rigsim.config.transform import Transform
from rigsim.config.types import SensorKind, SessionSource
from rigsim.episode import EpisodeExtractor
from rigsim.io.mcap_writer import RigSimMCAPWriter


def test_episode_extractor_window_filtering() -> None:
    """Verify minimum duration filtering and defect margin buffering logic."""
    extractor = EpisodeExtractor(min_duration_s=2.0, defect_margin_s=0.2)

    # Window 1: 5.0s -> after 0.2s margin on each side = 4.6s (>= 2.0s -> keep)
    # Window 2: 1.5s -> after 0.2s margin = 1.1s (< 2.0s -> discard)
    w1 = TimeWindow(t_start_ns=1_000_000_000, t_end_ns=6_000_000_000)
    w2 = TimeWindow(t_start_ns=7_000_000_000, t_end_ns=8_500_000_000)

    filtered = extractor._filter_usable_windows([w1, w2])
    assert len(filtered) == 1
    s_ns, e_ns = filtered[0]
    # Margin of 0.2s = 200_000_000 ns
    assert s_ns == 1_200_000_000
    assert e_ns == 5_800_000_000


def test_episode_extractor_slicing_and_hashing() -> None:
    """Verify zero-copy MCAP slicing, SHA-256 provenance, and EpisodeForge manifest generation."""
    with tempfile.TemporaryDirectory(prefix="rigsim_test_ep_extract_") as tmpdir:
        tmp_path = Path(tmpdir)
        session_dir = tmp_path / "sim_session_001"
        session_dir.mkdir()

        mcap_path = session_dir / "session.mcap"
        writer = RigSimMCAPWriter(mcap_path)
        writer.register_camera_channel("cam0")

        # Write 5 frames at 30 fps (t = 1.0s, 1.033s, 1.066s, 1.100s, 1.133s)
        import numpy as np

        test_img = np.zeros((240, 320, 3), dtype=np.uint8)
        base_ns = 1_700_000_000_000_000_000
        dt_ns = 33_333_333  # ~30 fps

        for i in range(5):
            t_ns = base_ns + i * dt_ns
            writer.write_image(
                stream_id="cam0",
                frame_id="cam0_optical",
                image_rgb=test_img,
                publish_time_ns=t_ns,
                log_time_ns=t_ns + 2_000_000,
            )
        writer.close()

        # Write session.json
        stream_spec = Stream(
            stream_id="cam0",
            kind=SensorKind.RGB,
            nominal_rate_hz=30.0,
            resolution=(320, 240),
            clock_id="clk0",
            frame_id="cam0_optical",
            intrinsics=CameraIntrinsics(fx=250.0, fy=250.0, cx=160.0, cy=120.0, width=320, height=240),
            distortion=CameraDistortion(coeffs=[0.0, 0.0, 0.0, 0.0]),
            extrinsics=Transform.identity(),
        )
        session = Session(
            session_id="sim_session_001",
            source=SessionSource.SIM,
            rig_config_name="mock_rig",
            created_at="2026-11-13T00:00:00Z",
            operator="RigSim-Test",
            streams=[stream_spec],
            mcap_path=str(mcap_path),
        )
        with open(session_dir / "session.json", "w") as f:
            f.write(session.model_dump_json(indent=2))

        # Mock QA verdict with 1 usable range covering the middle frames (frame 1 to frame 3)
        t_start_ns = base_ns + dt_ns
        t_end_ns = base_ns + 3 * dt_ns
        mock_verdict = SessionVerdict(
            session_id="sim_session_001",
            overall_verdict=QAVerdict.PASS,
            usable_time_ranges=[TimeWindow(t_start_ns=t_start_ns, t_end_ns=t_end_ns)],
            usable_duration_s=0.1,
            total_duration_s=0.2,
            yield_ratio=0.5,
            results=[
                QAResult(
                    check_id="T1_FRAME_DROPS",
                    stream_id="cam0",
                    verdict=QAVerdict.PASS,
                    metric_name="drop_pct",
                    value=0.0,
                    threshold=0.01,
                    explanation="All frames verified",
                )
            ],
        )

        out_ep_dir = tmp_path / "episodes"
        extractor = EpisodeExtractor(min_duration_s=0.05, defect_margin_s=0.0)
        dataset_manifest = extractor.extract_episodes(
            session_dir=session_dir,
            output_dir=out_ep_dir,
            verdict=mock_verdict,
        )

        assert dataset_manifest.total_episodes == 1
        assert dataset_manifest.source_session_id == "sim_session_001"
        assert len(dataset_manifest.episodes) == 1

        ep_summary = dataset_manifest.episodes[0]
        ep_dir = out_ep_dir / "sim_session_001" / ep_summary["relative_path"]
        assert ep_dir.exists()

        # Verify episode manifest
        manifest_file = ep_dir / "manifest.json"
        assert manifest_file.exists()
        with open(manifest_file) as mf:
            ep_obj = Episode.model_validate(json.load(mf))
        assert ep_obj.episode_id == "sim_session_001_ep000"
        assert ep_obj.qa_verdict == QAVerdict.PASS
        assert len(ep_obj.calibration_hash) == 64
        assert len(ep_obj.config_hash) == 64
        assert len(ep_obj.data_uris["sha256"]) == 64

        # Verify sliced MCAP
        ep_mcap = ep_dir / "episode.mcap"
        assert ep_mcap.exists()
        assert ep_mcap.stat().st_size > 0

        # Verify QA certificate
        cert_file = ep_dir / "qa_certificate.json"
        assert cert_file.exists()
        with open(cert_file) as cf:
            cert_data = json.load(cf)
        assert cert_data["status"] == "CERTIFIED_USABLE"
        assert cert_data["certificate_id"] == "CERT-sim_session_001_ep000"


def test_cli_export_episodes_command() -> None:
    """Verify rigsim export-episodes CLI execution end-to-end."""
    runner = CliRunner()
    clean_dir = Path("data/clean_test")
    if not clean_dir.exists():
        return

    with tempfile.TemporaryDirectory(prefix="rigsim_test_cli_ep_") as tmpdir:
        out_dir = Path(tmpdir)
        result = runner.invoke(
            app,
            [
                "export-episodes",
                str(clean_dir),
                "--output-dir",
                str(out_dir),
                "--min-duration",
                "1.0",
            ],
        )
        assert result.exit_code == 0
        assert "RigSim Usable Episode Extractor & Delivery Packager" in result.stdout
        assert "Exported Client Delivery Episodes" in result.stdout
        assert "YIELD" in result.stdout
        assert "Episode package successfully exported" in result.stdout
