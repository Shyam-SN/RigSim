"""Unit and integration tests for Milestone M8: Capture Planner, Bandwidth, Codecs & Costs."""

import tempfile
from pathlib import Path

import yaml
from typer.testing import CliRunner

from rigsim.cli import app
from rigsim.config.cost import StoragePricingConfig
from rigsim.config.rig import RigConfig
from rigsim.plan import (
    CodecBenchmarker,
    CodecResult,
    CostModel,
    compute_rig_bandwidth,
    generate_capture_planner_report,
)


def test_compute_rig_bandwidth_tabletop_6cam() -> None:
    """Verify raw bandwidth calculation and hardware interface bottleneck warnings on 6-cam + wrist rig."""
    rig_path = Path("configs/rigs/tabletop_6cam.yaml")
    assert rig_path.exists()

    with open(rig_path) as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    bw = compute_rig_bandwidth(rig_cfg)
    assert bw.rig_name == "tabletop_6cam"
    # 6 static cameras + 1 wrist camera = 7 cameras total
    assert len(bw.cameras) == 7
    assert bw.imu is not None

    # Static cameras: 1280x720 @ 30fps = 82.944 MB/s
    static_cams = [c for c in bw.cameras if c.stream_id != "cam_wrist"]
    for cam in static_cams:
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 30.0
        assert 82.0 < cam.raw_mb_per_s < 84.0

    # Wrist camera: 640x480 @ 60fps = 55.296 MB/s
    wrist_cam = next(c for c in bw.cameras if c.stream_id == "cam_wrist")
    assert wrist_cam.width == 640
    assert wrist_cam.height == 480
    assert wrist_cam.fps == 60.0
    assert 54.0 < wrist_cam.raw_mb_per_s < 56.0

    # Total 7 cameras + IMU: ~552.97 MB/s, ~1990 GB/hr
    assert 550.0 < bw.total_raw_mb_per_s < 555.0
    assert 1980.0 < bw.total_raw_gb_per_hour < 2000.0

    # 553 MB/s exceeds USB 3.0 (400 MB/s), 1GbE (110 MB/s), SATA SSD (450 MB/s), and Uplinks
    assert len(bw.warnings) >= 4
    assert any("USB 3.0" in w for w in bw.warnings)
    assert any("1 Gigabit Ethernet" in w for w in bw.warnings)
    assert any("SATA SSD" in w for w in bw.warnings)

    # NVMe M.2 (1500 MB/s) and 10GbE (1100 MB/s) should be safe
    assert bw.interface_comparisons["NVMe M.2 SSD (Sustained sequential write)"]["exceeded"] is False
    assert bw.interface_comparisons["10 Gigabit Ethernet (10GbE NIC)"]["exceeded"] is False


def test_compute_rig_bandwidth_handheld_stereo_imu() -> None:
    """Verify bandwidth on stereo rig fits USB 3.0, 1GbE, and SATA SSD."""
    rig_path = Path("configs/rigs/handheld_stereo_imu.yaml")
    assert rig_path.exists()

    with open(rig_path) as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    bw = compute_rig_bandwidth(rig_cfg)
    assert bw.rig_name == "handheld_stereo_imu"
    assert len(bw.cameras) == 2
    # 2 cams at 752x480 @ 20fps = ~43.33 MB/s
    assert 40.0 < bw.total_raw_mb_per_s < 46.0
    # Safe on USB 3.0 (400 MB/s limit), 1GbE (110 MB/s limit), and SATA SSD (450 MB/s limit)
    assert bw.interface_comparisons["USB 3.0 / 3.1 Gen 1 (Single Host Controller)"]["exceeded"] is False
    assert bw.interface_comparisons["1 Gigabit Ethernet (1GbE NIC)"]["exceeded"] is False
    assert bw.interface_comparisons["SATA SSD (Sustained sequential write)"]["exceeded"] is False
    # Exceeds USB 2.0 (35 MB/s limit) and 100Mbps Uplink (12.5 MB/s limit)
    assert bw.interface_comparisons["USB 2.0 (Bus limit)"]["exceeded"] is True
    assert bw.interface_comparisons["Internet Uplink (100 Mbps Facility Fiber)"]["exceeded"] is True


def test_codec_benchmarker_synthetic() -> None:
    """Test empirical CodecBenchmarker using an automated synthetic reference clip."""
    with tempfile.TemporaryDirectory(prefix="rigsim_test_bench_") as tmpdir:
        tmp_path = Path(tmpdir)
        clip_path = tmp_path / "synthetic_test.mp4"
        CodecBenchmarker.generate_synthetic_clip(
            clip_path, width=320, height=240, fps=15.0, duration_s=1.0
        )
        assert clip_path.exists()

        bench = CodecBenchmarker(working_dir=tmp_path / "work")
        results = bench.run_benchmark(clip_path, max_duration_s=1.0)

        # Should benchmark at least available software encoders (e.g. libx264, ffv1, png)
        assert len(results) >= 2
        for r in results:
            assert r.file_size_bytes > 0
            assert r.bitrate_kbps > 0
            assert r.compression_ratio >= 1.0
            assert r.encode_fps > 0
            assert 0.0 <= r.ssim <= 1.0
            assert r.psnr_db > 0
            assert 0.0 <= r.feature_retention_ratio <= 1.0

        # At least one result must be tagged Pareto-optimal
        assert any(r.is_pareto for r in results)
        bench.cleanup()


def test_cost_model_evaluation() -> None:
    """Verify multi-tier cloud storage cost modeling formulas."""
    rig_path = Path("configs/rigs/handheld_stereo_imu.yaml")
    with open(rig_path) as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    bw = compute_rig_bandwidth(rig_cfg)
    pricing = StoragePricingConfig(
        hot_storage_per_gb_month=0.023,
        cold_storage_per_gb_month=0.004,
        archive_storage_per_gb_month=0.00099,
    )
    cost_model = CostModel(pricing=pricing)
    report = cost_model.evaluate(bw, hours=100.0)

    assert report.hours_modeled == 100.0
    assert len(report.projections) >= 3

    raw_proj = report.projections[0]
    assert raw_proj.format_name.startswith("Raw")
    assert raw_proj.compression_ratio == 1.0

    # Test formula relations
    for proj in report.projections:
        assert abs(proj.storage_tb - (proj.storage_gb / 1000.0)) < 1e-4
        assert abs(proj.annual_hot_usd - (proj.monthly_hot_usd * 12.0)) < 1e-4
        assert abs(proj.annual_archive_usd - (proj.monthly_archive_usd * 12.0)) < 1e-4
        assert abs(proj.three_year_archive_usd - (proj.annual_archive_usd * 3.0)) < 1e-4
        assert proj.upload_duration_hours_100mbps > proj.upload_duration_hours_1gbps
        assert proj.upload_duration_hours_1gbps > proj.upload_duration_hours_10gbps

    # Compressed projections must show cost savings compared to raw
    for proj in report.projections[1:]:
        assert proj.storage_tb < raw_proj.storage_tb
        assert proj.annual_savings_vs_raw_hot_usd > 0.0
        assert proj.annual_savings_vs_raw_archive_usd > 0.0


def test_generate_capture_planner_report() -> None:
    """Verify HTML capture planner report generation with Plotly figures."""
    rig_path = Path("configs/rigs/tabletop_6cam.yaml")
    with open(rig_path) as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    bw = compute_rig_bandwidth(rig_cfg)
    cost_model = CostModel()
    cost_report = cost_model.evaluate(bw, hours=500.0)

    mock_codec_results = [
        CodecResult(
            codec_name="libx264",
            preset_name="x264 CRF 23",
            file_size_bytes=5_000_000,
            bitrate_kbps=4000.0,
            compression_ratio=11.2,
            encode_time_s=1.2,
            encode_fps=75.0,
            speedup_realtime=2.5,
            ssim=0.985,
            psnr_db=42.1,
            feature_retention_ratio=0.97,
            is_pareto=True,
        ),
        CodecResult(
            codec_name="libx265",
            preset_name="x265 CRF 28",
            file_size_bytes=3_500_000,
            bitrate_kbps=2800.0,
            compression_ratio=16.0,
            encode_time_s=2.5,
            encode_fps=36.0,
            speedup_realtime=1.2,
            ssim=0.981,
            psnr_db=40.5,
            feature_retention_ratio=0.95,
            is_pareto=True,
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="rigsim_test_report_") as tmpdir:
        report_file = Path(tmpdir) / "plan_report.html"
        generated = generate_capture_planner_report(
            bandwidth=bw,
            cost_report=cost_report,
            codec_results=mock_codec_results,
            output_path=report_file,
        )
        assert generated.exists()
        html = generated.read_text(encoding="utf-8")
        assert "Capture Planner & Economics Report" in html
        assert "tabletop_6cam" in html
        assert "Rate-Distortion Efficiency" in html
        assert "Pareto Optimal" in html
        assert "Cloud Storage, Data Transfer & Cost Projections" in html


def test_cli_plan_command() -> None:
    """Test rigsim plan command in CLI runner."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory(prefix="rigsim_test_cli_") as tmpdir:
        out_html = Path(tmpdir) / "test_plan.html"
        result = runner.invoke(
            app,
            [
                "plan",
                "configs/rigs/handheld_stereo_imu.yaml",
                "--skip-bench",
                "--output",
                str(out_html),
                "--hours",
                "250",
            ],
        )
        assert result.exit_code == 0
        assert "Rig Sensor Bandwidth Breakdown" in result.stdout
        assert "Physical Bus & Transport Interface Headroom" in result.stdout
        assert "Cloud Storage & Transfer Projections" in result.stdout
        assert out_html.exists()
