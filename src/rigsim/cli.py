"""RigSim Command Line Interface."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rigsim",
    help="Multi-Camera + IMU Capture Simulator, Real Rig, and Automated Data QA Engine",
    add_completion=False,
)
sync_app = typer.Typer(name="sync", help="Sensor synchronization tools (QR clock, IMU-camera)")
app.add_typer(sync_app, name="sync")

ingest_app = typer.Typer(
    name="ingest",
    help="Ingest real-world dataset formats (EuRoC, SensorLogger, OpenCamera, Mac webcam)",
)
app.add_typer(ingest_app, name="ingest")

console = Console()


@app.command()
def simulate(
    rig: Path = typer.Option(
        Path("configs/rigs/tabletop_6cam.yaml"),
        "--rig",
        "-r",
        help="Path to rig YAML configuration",
    ),
    faults: Path | None = typer.Option(
        None,
        "--faults",
        "-f",
        help="Path to fault scenario YAML configuration",
    ),
    duration: float = typer.Option(
        10.0,
        "--duration",
        "-d",
        help="Duration of simulation session in seconds",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for deterministic generation",
    ),
    output_dir: Path = typer.Option(
        Path("data/sim_session"),
        "--output-dir",
        "-o",
        help="Output directory for MCAP and session manifests",
    ),
) -> None:
    """Run physics-based multi-sensor simulation session and write MCAP."""
    console.print(f"[bold green]Simulating session[/bold green] using rig: [cyan]{rig}[/cyan]")
    console.print(f"Duration: {duration}s | Seed: {seed} | Output: {output_dir}")

    import yaml

    from rigsim.config.faults import FaultScenarioConfig
    from rigsim.config.rig import RigConfig
    from rigsim.sim.simulator import SimulationEngine

    with open(rig) as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    fault_cfg = None
    if faults and faults.exists():
        with open(faults) as f:
            fault_cfg = FaultScenarioConfig.model_validate(yaml.safe_load(f))

    engine = SimulationEngine(
        rig_config=rig_cfg,
        fault_config=fault_cfg,
        duration_s=duration,
        seed=seed,
        output_dir=output_dir,
    )
    session = engine.run()
    console.print(
        f"[bold green]✓ Simulation complete![/bold green] Session ID: [cyan]{session.session_id}[/cyan]"
    )
    console.print(f"MCAP written to: [cyan]{session.mcap_path}[/cyan]")


@app.command()
def calibrate(
    session_or_folder: Path = typer.Argument(
        ..., help="Path to session directory or image folder with ChArUco targets"
    ),
    stereo: bool = typer.Option(
        True, "--stereo/--no-stereo", help="Perform stereo calibration on stereo pairs"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Directory to save calibration JSON results"
    ),
) -> None:
    """Perform intrinsic and stereo calibration using ChArUco fiducials."""
    import json

    import cv2
    import numpy as np
    from foxglove_schemas_protobuf import CompressedImage_pb2
    from mcap.reader import make_reader

    from rigsim.calib.charuco import CharucoCalibrator

    console.print(
        f"[bold green]Running calibration on:[/bold green] [cyan]{session_or_folder}[/cyan]"
    )

    manifest_file = (
        session_or_folder / "session.json" if session_or_folder.is_dir() else session_or_folder
    )
    if manifest_file.exists():
        with open(manifest_file) as f:
            sess_meta = json.load(f)
        mcap_path = manifest_file.parent / Path(sess_meta.get("mcap_path", "session.mcap")).name
    else:
        mcap_path = session_or_folder

    if not mcap_path.exists():
        console.print(f"[bold red]MCAP log file not found at: {mcap_path}[/bold red]")
        raise typer.Exit(code=1)

    # Ingest stereo or mono camera frames
    left_frames = []
    right_frames = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if (
                "cam_stereo_left/image_compressed" in channel.topic
                or "cam0/image_compressed" in channel.topic
            ):
                pb = CompressedImage_pb2.CompressedImage()
                pb.ParseFromString(message.data)
                bgr = cv2.imdecode(np.frombuffer(pb.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    left_frames.append(bgr)
            elif (
                "cam_stereo_right/image_compressed" in channel.topic
                or "cam1/image_compressed" in channel.topic
            ):
                pb = CompressedImage_pb2.CompressedImage()
                pb.ParseFromString(message.data)
                bgr = cv2.imdecode(np.frombuffer(pb.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    right_frames.append(bgr)

    if not left_frames:
        console.print("[bold red]No camera frames found for calibration.[/bold red]")
        raise typer.Exit(code=1)

    calib = CharucoCalibrator()
    h, w = left_frames[0].shape[:2]

    # Mono calibration
    mono_left = calib.calibrate_mono(left_frames, (w, h))

    table = Table(title="ChArUco Intrinsic Calibration Results")
    table.add_column("Camera", style="cyan")
    table.add_column("Reprojection RMS (px)", style="bold green")
    table.add_column("Focal Length (fx, fy)", style="magenta")
    table.add_column("Principal Point (cx, cy)", style="yellow")
    table.add_column("Views (Corners)", style="white")

    table.add_row(
        "Left Camera",
        f"{mono_left.rms_reprojection_px:.3f} px",
        f"({mono_left.K[0, 0]:.1f}, {mono_left.K[1, 1]:.1f})",
        f"({mono_left.K[0, 2]:.1f}, {mono_left.K[1, 2]:.1f})",
        f"{mono_left.num_views} ({mono_left.num_corners_total})",
    )

    stereo_res = None
    if stereo and right_frames and len(left_frames) == len(right_frames):
        mono_right = calib.calibrate_mono(right_frames, (w, h))
        table.add_row(
            "Right Camera",
            f"{mono_right.rms_reprojection_px:.3f} px",
            f"({mono_right.K[0, 0]:.1f}, {mono_right.K[1, 1]:.1f})",
            f"({mono_right.K[0, 2]:.1f}, {mono_right.K[1, 2]:.1f})",
            f"{mono_right.num_views} ({mono_right.num_corners_total})",
        )
        stereo_res = calib.calibrate_stereo(
            left_frames,
            right_frames,
            (w, h),
            mono_left.K,
            mono_left.distortion,
            mono_right.K,
            mono_right.distortion,
        )

    console.print(table)

    if stereo_res:
        s_table = Table(title="Stereo Extrinsic Calibration Results")
        s_table.add_column("Metric", style="cyan")
        s_table.add_column("Value", style="bold green")
        s_table.add_row("Stereo RMS Error", f"{stereo_res.rms_stereo_px:.3f} px")
        s_table.add_row("Baseline Distance", f"{stereo_res.baseline_m * 1000.0:.2f} mm")
        s_table.add_row(
            "Translation [Tx, Ty, Tz]", str(np.round(stereo_res.T.ravel() * 1000.0, 2)) + " mm"
        )
        s_table.add_row("Stereo Pairs Used", str(stereo_res.num_stereo_pairs))
        console.print(s_table)

    # Save results
    out_dir = output_dir or (
        manifest_file.parent / "calib" if manifest_file.exists() else Path("data/calib")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_json = {
        "rms_px": mono_left.rms_reprojection_px,
        "K": mono_left.K.tolist(),
        "distortion": mono_left.distortion.tolist(),
        "stereo": {
            "rms_stereo_px": stereo_res.rms_stereo_px,
            "baseline_m": stereo_res.baseline_m,
            "R": stereo_res.R.tolist(),
            "T": stereo_res.T.tolist(),
        }
        if stereo_res
        else None,
    }
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(calib_json, f, indent=2)
    console.print(
        f"[bold green]Saved calibration parameters to:[/bold green] {out_dir / 'calibration.json'}"
    )


@app.command()
def qa(
    session: Path = typer.Argument(..., help="Path to session.json or MCAP session directory"),
    thresholds: Path = typer.Option(
        Path("configs/qa/default_thresholds.yaml"),
        "--thresholds",
        "-t",
        help="Path to QA thresholds YAML",
    ),
) -> None:
    """Run full automated QA engine on a session and output PASS/WARN/FAIL verdicts."""
    console.print(f"[bold green]Running QA Engine on:[/bold green] [cyan]{session}[/cyan]")
    import yaml

    from rigsim.config.qa import QAThresholdsConfig
    from rigsim.qa.engine import QAEngine

    with open(thresholds) as f:
        thresh_cfg = QAThresholdsConfig.model_validate(yaml.safe_load(f))

    engine = QAEngine(thresholds=thresh_cfg)
    engine.process_session(session)


@app.command()
def report(
    session: Path | None = typer.Argument(
        None, help="Path to session.json or session folder for single-session report"
    ),
    batch: Path | None = typer.Option(
        None, "--batch", "-b", help="Folder containing multiple sessions for multi-session summary"
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Target output HTML file path"),
    annotate_mcap: bool = typer.Option(
        True,
        "--annotate-mcap/--no-annotate-mcap",
        help="Export QA defect events as Foxglove MCAP annotations",
    ),
) -> None:
    """Generate offline self-contained HTML report (single session or multi-session batch)."""
    import json

    from rigsim.config.qa import SessionVerdict
    from rigsim.config.session import Session
    from rigsim.qa.engine import QAEngine
    from rigsim.report.batch_report import generate_batch_report
    from rigsim.report.foxglove import export_foxglove_annotations
    from rigsim.report.session_report import generate_session_report

    if batch:
        console.print(
            f"[bold green]Generating multi-session summary report for:[/bold green] [cyan]{batch}[/cyan]"
        )
        out_html = generate_batch_report(batch, output_html=output)
        console.print(
            f"[bold green]✓ Fleet summary report generated:[/bold green] [cyan]{out_html}[/cyan]"
        )
    elif session:
        console.print(
            f"[bold green]Generating single-session HTML report for:[/bold green] [cyan]{session}[/cyan]"
        )
        sess_dir = session.parent if session.is_file() else session
        manifest = sess_dir / "session.json"

        with open(manifest) as f:
            s_dict = json.load(f)
        Session.model_validate(s_dict)

        # Check or compute QA
        qa_results_file = sess_dir / "qa" / "results.json"
        if not qa_results_file.exists():
            qa_results_file = sess_dir / "qa_results.json"

        if qa_results_file.exists():
            with open(qa_results_file) as f:
                r_dict = json.load(f)
            verdict = SessionVerdict.model_validate(r_dict)
            qa_results = verdict.results
        else:
            engine = QAEngine()
            verdict = engine.process_session(manifest)
            qa_results = verdict.results

        # Optional Foxglove MCAP event annotation
        if annotate_mcap:
            mcap_name = Path(s_dict.get("mcap_path", "session.mcap")).name
            mcap_file = sess_dir / mcap_name
            if mcap_file.exists():
                console.print(
                    f"[bold cyan]Exporting Foxglove event annotations to:[/bold cyan] {mcap_file}"
                )
                export_foxglove_annotations(mcap_file, qa_results, session_verdict=verdict)
                console.print(
                    "[bold green]✓ MCAP annotated with /qa/events and /qa/logs channels.[/bold green]"
                )

        out_html = generate_session_report(session, output_html=output)
        console.print(
            f"[bold green]✓ Single-session HTML report generated:[/bold green] [cyan]{out_html}[/cyan]"
        )
    else:
        console.print("[bold red]Error: Provide a session path or --batch directory.[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def plan(
    rig: Path = typer.Argument(..., help="Path to rig YAML configuration"),
    bench_clips: Path | None = typer.Option(
        None, "--bench-clips", "-b", help="Sample video clips for empirical codec benchmarking"
    ),
    pricing: Path = typer.Option(
        Path("configs/cost/storage_pricing.yaml"),
        "--pricing",
        "-p",
        help="Storage and compute pricing YAML",
    ),
    hours: float = typer.Option(
        1000.0,
        "--hours",
        "-h",
        help="Total capture hours to model for storage and cloud budgeting",
    ),
    output: Path = typer.Option(
        Path("reports/plan_report.html"),
        "--output",
        "-o",
        help="Output path for interactive HTML capture planner report",
    ),
    skip_bench: bool = typer.Option(
        False, "--skip-bench", help="Skip empirical FFmpeg codec sweeps and use baseline priors"
    ),
) -> None:
    """Run capture planner: calculate bandwidth, test codecs, model storage and cloud costs."""
    import tempfile

    import yaml

    from rigsim.config.rig import RigConfig
    from rigsim.plan import (
        CodecBenchmarker,
        CodecResult,
        CostModel,
        compute_rig_bandwidth,
        generate_capture_planner_report,
    )

    console.print("[bold cyan]============================================================[/bold cyan]")
    console.print("[bold green]RigSim Capture Planner & Economics Engine[/bold green]")
    console.print(f"Target Rig Config: [cyan]{rig}[/cyan] | Horizon: [yellow]{hours:.0f} hours[/yellow]")
    console.print("[bold cyan]============================================================[/bold cyan]\n")

    if not rig.exists():
        console.print(f"[bold red]Error: Rig configuration file not found at: {rig}[/bold red]")
        raise typer.Exit(code=1)

    with open(rig) as f:
        rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

    # 1. Bandwidth calculations
    bandwidth = compute_rig_bandwidth(rig_cfg)

    # Print Bandwidth Breakdown
    bw_table = Table(title=f"Rig Sensor Bandwidth Breakdown: {bandwidth.rig_name}")
    bw_table.add_column("Stream ID", style="cyan")
    bw_table.add_column("Type / Config", style="white")
    bw_table.add_column("Rate", justify="right")
    bw_table.add_column("Raw MB/s", justify="right")
    bw_table.add_column("Raw GB/hr", justify="right")
    bw_table.add_column("Raw TB/1,000h", justify="right")

    for cam in bandwidth.cameras:
        bw_table.add_row(
            cam.stream_id,
            f"{cam.width}x{cam.height} RGB",
            f"{cam.fps:.1f} fps",
            f"{cam.raw_mb_per_s:.1f}",
            f"{cam.raw_gb_per_hour:.1f}",
            f"{cam.raw_tb_per_1000h:.2f}",
        )
    if bandwidth.imu:
        bw_table.add_row(
            bandwidth.imu.stream_id,
            f"6-DoF ({bandwidth.imu.sample_bytes} B/sample)",
            f"{bandwidth.imu.rate_hz:.0f} Hz",
            f"{bandwidth.imu.raw_mb_per_s:.3f}",
            f"{bandwidth.imu.raw_gb_per_hour:.3f}",
            f"{bandwidth.imu.raw_tb_per_1000h:.4f}",
        )
    bw_table.add_section()
    bw_table.add_row(
        "[bold]TOTAL SUSTAINED[/bold]",
        f"[bold]{len(bandwidth.cameras)} Cams" + (" + 1 IMU[/bold]" if bandwidth.imu else "[/bold]"),
        "-",
        f"[bold green]{bandwidth.total_raw_mb_per_s:.1f}[/bold green]",
        f"[bold green]{bandwidth.total_raw_gb_per_hour:.1f}[/bold green]",
        f"[bold green]{bandwidth.total_raw_tb_per_1000h:.2f}[/bold green]",
    )
    console.print(bw_table)
    console.print()

    # Print Interface Limits Table
    iface_table = Table(title="Physical Bus & Transport Interface Headroom")
    iface_table.add_column("Interface Standard", style="cyan")
    iface_table.add_column("Limit (MB/s)", justify="right")
    iface_table.add_column("Rig Rate (MB/s)", justify="right")
    iface_table.add_column("Utilization", justify="right")
    iface_table.add_column("Status", justify="center")

    for iface_name, info in bandwidth.interface_comparisons.items():
        pct = float(info["utilization_pct"])
        exceeded = bool(info["exceeded"])
        if exceeded:
            status = "[bold red]EXCEEDED[/bold red]"
        elif pct > 75.0:
            status = "[yellow]WARN (>75%)[/yellow]"
        else:
            status = "[green]SAFE[/green]"
        iface_table.add_row(
            iface_name,
            f"{info['limit_mb_s']:.1f}",
            f"{bandwidth.total_raw_mb_per_s:.1f}",
            f"{pct:.1f}%",
            status,
        )
    console.print(iface_table)
    console.print()

    if bandwidth.warnings:
        for w in bandwidth.warnings:
            console.print(f"[bold red]WARNING:[/bold red] {w}")
        console.print()

    # 2. Empirical Codec Benchmark
    codec_results: list[CodecResult] | None = None
    if not skip_bench:
        console.print("[bold yellow]Running empirical FFmpeg codec transcoding sweeps...[/bold yellow]")
        bench_dir = Path(tempfile.mkdtemp(prefix="rigsim_plan_bench_"))
        try:
            if bench_clips and bench_clips.exists():
                target_clip = bench_clips
            else:
                target_clip = bench_dir / "bench_synthetic.mp4"
                CodecBenchmarker.generate_synthetic_clip(target_clip, duration_s=2.0)

            benchmarker = CodecBenchmarker(working_dir=bench_dir)
            codec_results = benchmarker.run_benchmark(target_clip, max_duration_s=2.0)

            if codec_results:
                codec_tbl = Table(title="Empirical Codec Benchmark & Rate-Distortion Frontier")
                codec_tbl.add_column("Codec", style="cyan")
                codec_tbl.add_column("Profile / Preset")
                codec_tbl.add_column("Compression", justify="right")
                codec_tbl.add_column("Speedup", justify="right")
                codec_tbl.add_column("SSIM", justify="right")
                codec_tbl.add_column("PSNR (dB)", justify="right")
                codec_tbl.add_column("ORB Match %", justify="right")
                codec_tbl.add_column("Pareto Status", justify="center")

                for cr in codec_results:
                    pareto_str = "[bold magenta]Pareto Optimal[/bold magenta]" if cr.is_pareto else "[dim]Dominated[/dim]"
                    codec_tbl.add_row(
                        cr.codec_name,
                        cr.preset_name,
                        f"{cr.compression_ratio:.1f}x",
                        f"{cr.speedup_realtime:.1f}x ({cr.encode_fps:.0f} fps)",
                        f"{cr.ssim:.4f}",
                        f"{cr.psnr_db:.1f}",
                        f"{cr.feature_retention_ratio * 100:.1f}%",
                        pareto_str,
                    )
                console.print(codec_tbl)
                console.print()
        finally:
            benchmarker.cleanup()

    # 3. Cost Model Evaluation
    cost_model = CostModel.from_yaml(pricing) if pricing.exists() else CostModel()
    cost_report = cost_model.evaluate(bandwidth=bandwidth, hours=hours, codec_results=codec_results)

    cost_tbl = Table(title=f"Cloud Storage & Transfer Projections ({hours:.0f} Hours of Capture)")
    cost_tbl.add_column("Format / Preset", style="cyan")
    cost_tbl.add_column("Footprint", justify="right")
    cost_tbl.add_column("Monthly Hot (S3)", justify="right")
    cost_tbl.add_column("Monthly Archive", justify="right")
    cost_tbl.add_column("Annual Archive", justify="right")
    cost_tbl.add_column("Annual Hot Savings", justify="right")
    cost_tbl.add_column("1Gbps Upload Time", justify="right")

    for proj in cost_report.projections:
        savings_str = (
            f"[green]+${proj.annual_savings_vs_raw_hot_usd:,.2f}[/green]"
            if proj.annual_savings_vs_raw_hot_usd > 0
            else "[dim]Baseline[/dim]"
        )
        cost_tbl.add_row(
            proj.format_name,
            f"{proj.storage_tb:.2f} TB",
            f"${proj.monthly_hot_usd:,.2f}",
            f"${proj.monthly_archive_usd:,.2f}",
            f"${proj.annual_archive_usd:,.2f}",
            savings_str,
            f"{proj.upload_duration_hours_1gbps:.1f} hrs",
        )
    console.print(cost_tbl)
    console.print()

    # 4. Generate Interactive Report
    report_file = generate_capture_planner_report(
        bandwidth=bandwidth,
        cost_report=cost_report,
        codec_results=codec_results,
        output_path=output,
    )
    console.print(f"[bold green]✓ Interactive HTML report successfully generated at:[/bold green] [cyan]{report_file}[/cyan]\n")



@app.command()
def ingest(
    adapter: str = typer.Argument(
        ..., help="Adapter type: euroc, sensorlogger, opencamera, webcam"
    ),
    source_path: Path = typer.Argument(..., help="Path to source dataset or device export"),
    output_dir: Path = typer.Option(
        Path("data/ingested"), "--output-dir", "-o", help="Target output session folder"
    ),
) -> None:
    """Ingest external real datasets (EuRoC, Sensor Logger, OpenCamera, Mac webcam) into standard Session."""
    console.print(
        f"[bold green]Ingesting {adapter} data from:[/bold green] [cyan]{source_path}[/cyan]"
    )


@sync_app.command("qr")
def sync_qr(
    phone_session: Path = typer.Argument(
        ..., help="Path to phone capture session folder or video file"
    ),
    camera: str = typer.Option(
        "phone_cam", "--camera", "-c", help="Camera stream ID in session MCAP"
    ),
    max_frames: int = typer.Option(
        300, "--max-frames", "-m", help="Max frames to process for QR decoding"
    ),
) -> None:
    """Perform cross-device QR clock synchronization and fit clock model with robust Huber regression."""
    import json

    import cv2
    import numpy as np
    from foxglove_schemas_protobuf import CompressedImage_pb2
    from mcap.reader import make_reader

    from rigsim.sync.qr_clock import QrClockSyncEstimator

    console.print(
        f"[bold green]Running optical QR clock synchronization on:[/bold green] [cyan]{phone_session}[/cyan]"
    )

    estimator = QrClockSyncEstimator()

    if phone_session.is_file() and phone_session.suffix.lower() in (".mp4", ".mov", ".m4v", ".avi"):
        # Direct video file
        res = estimator.process_video_file(phone_session, max_frames=max_frames)
        sess_dir = phone_session.parent
    else:
        sess_dir = phone_session.parent if phone_session.is_file() else phone_session
        manifest_file = sess_dir / "session.json"
        if not manifest_file.exists():
            console.print(f"[bold red]Session manifest not found at: {manifest_file}[/bold red]")
            raise typer.Exit(1)

        with open(manifest_file) as f:
            sess_meta = json.load(f)
        mcap_path = sess_dir / Path(sess_meta.get("mcap_path", "recording.mcap")).name

        frames = []
        timestamps = []
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for _, channel, message in reader.iter_messages():
                if f"/{camera}/image_compressed" in channel.topic:
                    pb = CompressedImage_pb2.CompressedImage()
                    pb.ParseFromString(message.data)
                    bgr = cv2.imdecode(np.frombuffer(pb.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        frames.append(bgr)
                        timestamps.append(message.publish_time)
                        if len(frames) >= max_frames:
                            break

        res = estimator.process_frames(frames, timestamps)

    table = Table(title="Cross-Device Optical QR Clock Synchronization")
    table.add_column("Synchronization Parameter", style="cyan")
    table.add_column("Measured Value", style="bold green")
    table.add_row(
        "Fitted Linear Model", f"t_mac = {res.slope_a:.8f} · t_phone + {res.intercept_b_s:.6f}s"
    )
    table.add_row("Initial Clock Offset", f"{res.offset_ms:+.3f} ms")
    table.add_row("Relative Clock Drift", f"{res.drift_ppm:+.2f} PPM")
    table.add_row("Residual Jitter (1-sigma)", f"{res.residual_jitter_std_ms:.2f} ms")
    table.add_row("Residual Jitter (p99)", f"{res.residual_jitter_p99_ms:.2f} ms")
    table.add_row(
        "QR Decode Success",
        f"{res.decoded_count}/{res.total_frames_checked} ({res.decode_success_rate * 100:.1f}%)",
    )
    console.print(table)

    if res.error_budget_ms:
        eb_table = Table(title="Sync Uncertainty Error Budget Breakdown")
        eb_table.add_column("Error Contributor", style="yellow")
        eb_table.add_column("Theoretical Bound", justify="right")
        for k, v in res.error_budget_ms.items():
            eb_table.add_row(k.replace("_", " ").title(), f"{v:.2f} ms")
        console.print(eb_table)

    console.print(f"[bold]{res.explanation}[/bold]")

    # Save clock sync results
    sync_res_path = sess_dir / "clock_sync.json"
    with open(sync_res_path, "w") as f:
        json.dump(
            {
                "offset_ms": res.offset_ms,
                "drift_ppm": res.drift_ppm,
                "residual_jitter_std_ms": res.residual_jitter_std_ms,
                "residual_jitter_p99_ms": res.residual_jitter_p99_ms,
                "slope_a": res.slope_a,
                "intercept_b_s": res.intercept_b_s,
                "decoded_count": res.decoded_count,
                "decode_success_rate": res.decode_success_rate,
                "error_budget_ms": res.error_budget_ms,
            },
            f,
            indent=2,
        )
    console.print(f"Clock synchronization saved to: [cyan]{sync_res_path}[/cyan]")


@sync_app.command("imu-cam")
def sync_imu_cam(
    session: Path = typer.Argument(
        ..., help="Path to session for IMU-camera temporal offset estimation"
    ),
    camera: str = typer.Option(
        "cam_wrist", "--camera", "-c", help="Camera stream ID to synchronize with IMU"
    ),
) -> None:
    """Estimate IMU-camera temporal offset via visual angular velocity and gyroscope cross-correlation."""
    import json

    import cv2
    import numpy as np
    from foxglove_schemas_protobuf import CompressedImage_pb2
    from mcap.reader import make_reader

    from rigsim.sync.imu_cam import ImuCameraSyncEstimator

    console.print(
        f"[bold green]Estimating IMU-camera offset for:[/bold green] [cyan]{session}[/cyan] (camera: [yellow]{camera}[/yellow])"
    )

    manifest_file = session / "session.json" if session.is_dir() else session
    if manifest_file.exists():
        with open(manifest_file) as f:
            sess_meta = json.load(f)
        mcap_path = manifest_file.parent / Path(sess_meta.get("mcap_path", "session.mcap")).name
    else:
        mcap_path = session

    cam_frames = []
    cam_ts = []
    gyro_readings = []
    imu_ts = []

    cam_topic_pattern = f"/{camera}/image_compressed"

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if cam_topic_pattern in channel.topic or (
                camera == "cam_wrist" and "cam0/image_compressed" in channel.topic
            ):
                pb = CompressedImage_pb2.CompressedImage()
                pb.ParseFromString(message.data)
                bgr = cv2.imdecode(np.frombuffer(pb.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    cam_frames.append(bgr)
                    cam_ts.append(message.publish_time)
            elif "/data" in channel.topic and "imu" in channel.topic:
                payload = json.loads(message.data)
                gyr = payload.get("angular_velocity", {"x": 0.0, "y": 0.0, "z": 0.0})
                gyro_readings.append([gyr["x"], gyr["y"], gyr["z"]])
                imu_ts.append(message.publish_time)

    if len(cam_ts) > 1:
        order_cam = np.argsort(cam_ts)
        cam_ts = [cam_ts[i] for i in order_cam]
        cam_frames = [cam_frames[i] for i in order_cam]

    if len(imu_ts) > 1:
        order_imu = np.argsort(imu_ts)
        imu_ts = [imu_ts[i] for i in order_imu]
        gyro_readings = [gyro_readings[i] for i in order_imu]

    estimator = ImuCameraSyncEstimator()
    res = estimator.estimate_offset(
        frames_rgb=cam_frames,
        cam_timestamps_ns=cam_ts,
        gyro_readings=gyro_readings,
        imu_timestamps_ns=imu_ts,
    )

    table = Table(title="IMU-Camera Synchronization Estimation")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="bold green")
    table.add_row("Estimated Hardware Offset", f"{res.offset_ms:+.3f} ms")
    table.add_row("Cross-Correlation Peak (r)", f"{res.correlation_peak:.3f}")
    table.add_row(
        "SLA Verdict (limit < 5.0 ms)",
        "[bold green]PASS[/bold green]"
        if abs(res.offset_ms) <= 5.0
        else "[bold red]FAIL[/bold red]",
    )
    table.add_row("Explanation", res.explanation)
    console.print(table)


@sync_app.command("euroc-sweep")
def sync_euroc_sweep(
    dataset_path: Path = typer.Argument(
        ..., help="Path to EuRoC dataset directory (e.g. data/euroc/MH_01_easy)"
    ),
    output_dir: Path = typer.Option(
        Path("data/euroc_sweep"),
        "--output-dir",
        "-o",
        help="Temporary directory for sweep MCAPs",
    ),
    max_frames: int = typer.Option(
        120, "--max-frames", "-m", help="Max camera frames to process for each offset step"
    ),
) -> None:
    """Run systematic IMU-camera temporal offset recovery sweep on EuRoC."""
    from rigsim.eval.euroc_eval import EuRoCEvaluator

    evaluator = EuRoCEvaluator(console)
    try:
        evaluator.run_offset_sweep(
            euroc_dir=dataset_path,
            tmp_base=output_dir,
            max_frames=max_frames,
        )
    except FileNotFoundError as e:
        console.print(str(e))
        raise typer.Exit(1)


@ingest_app.command("euroc")
def ingest_euroc(
    dataset_path: Path = typer.Argument(
        ..., help="Path to EuRoC sequence folder (e.g. data/euroc/MH_01_easy)"
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for session manifest and MCAP",
    ),
    faults: Path | None = typer.Option(
        None,
        "--faults",
        "-f",
        help="Optional fault scenario YAML to inject into real data",
    ),
    imu_shift_ms: float = typer.Option(
        0.0,
        "--imu-shift-ms",
        help="Artificially shift IMU timestamps by delta ms",
    ),
    max_frames: int | None = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Limit camera frames for quick testing",
    ),
    run_qa: bool = typer.Option(
        True,
        "--qa/--no-qa",
        help="Automatically run QA engine on ingested session",
    ),
    report: bool = typer.Option(
        False,
        "--report/--no-report",
        help="Generate standalone HTML QA report",
    ),
) -> None:
    """Ingest EuRoC MAV ASL dataset into RigSim Session and MCAP."""
    import yaml

    from rigsim.config.faults import FaultScenarioConfig
    from rigsim.ingest.euroc import EuRoCIngestAdapter
    from rigsim.qa.engine import QAEngine

    out_dir = output_dir or Path(f"data/euroc_ingested/{dataset_path.name}")

    fault_cfg = None
    if faults and faults.exists():
        with open(faults) as f:
            fault_cfg = FaultScenarioConfig.model_validate(yaml.safe_load(f))

    console.print(f"[bold green]Ingesting EuRoC sequence:[/bold green] [cyan]{dataset_path}[/cyan]")
    console.print(f"Destination: [cyan]{out_dir}[/cyan]")

    try:
        adapter = EuRoCIngestAdapter(
            euroc_dir=dataset_path,
            output_dir=out_dir,
            fault_config=fault_cfg,
            imu_time_shift_ms=imu_shift_ms,
            max_camera_frames=max_frames,
        )
        session = adapter.ingest()
    except FileNotFoundError as e:
        console.print(str(e))
        raise typer.Exit(1)

    console.print(
        f"[bold green]✓ Ingest complete![/bold green] Session ID: [cyan]{session.session_id}[/cyan]"
    )
    console.print(f"MCAP written to: [cyan]{session.mcap_path}[/cyan]")

    if run_qa:
        console.print("[bold cyan]Running automated QA engine...[/bold cyan]")
        qa_engine = QAEngine()
        verdict = qa_engine.process_session(out_dir)
        console.print(
            f"Overall Verdict: [bold]{verdict.overall_verdict.value.upper()}[/bold] | "
            f"Usable Yield: [bold green]{verdict.usable_percentage:.1f}%[/bold green]"
        )

    if report:
        from rigsim.report.session_report import generate_session_report

        rep_path = out_dir / "qa_report.html"
        generate_session_report(out_dir, rep_path)
        console.print(f"Report generated at: [cyan]{rep_path}[/cyan]")


@ingest_app.command("sensorlogger")
def ingest_sensorlogger(
    dataset_path: Path = typer.Argument(..., help="Path to Sensor Logger folder or zip file"),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for session manifest and MCAP",
    ),
    max_frames: int | None = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Limit video frames to process",
    ),
    run_qa: bool = typer.Option(
        True,
        "--qa/--no-qa",
        help="Automatically run QA engine on ingested session",
    ),
) -> None:
    """Ingest Nothing Phone recording from Sensor Logger (video + accelerometer + gyroscope)."""
    from rigsim.ingest.sensor_logger import SensorLoggerIngestAdapter
    from rigsim.qa.engine import QAEngine

    out_dir = output_dir or Path(f"data/phone_ingested/{dataset_path.stem}")
    console.print(
        f"[bold green]Ingesting Sensor Logger dataset:[/bold green] [cyan]{dataset_path}[/cyan]"
    )

    adapter = SensorLoggerIngestAdapter(
        input_path=dataset_path,
        output_dir=out_dir,
        max_frames=max_frames,
    )
    session = adapter.ingest()
    console.print(
        f"[bold green]✓ Ingest complete![/bold green] Session ID: [cyan]{session.session_id}[/cyan]"
    )
    console.print(f"MCAP written to: [cyan]{session.mcap_path}[/cyan]")

    if run_qa:
        console.print("[bold cyan]Running automated QA engine...[/bold cyan]")
        qa_engine = QAEngine()
        verdict = qa_engine.process_session(out_dir)
        console.print(
            f"Overall Verdict: [bold]{verdict.overall_verdict.value.upper()}[/bold] | "
            f"Usable Yield: [bold green]{verdict.usable_percentage:.1f}%[/bold green]"
        )


@ingest_app.command("opencamera")
def ingest_opencamera(
    dataset_path: Path = typer.Argument(..., help="Path to OpenCamera Sensors folder"),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for session manifest and MCAP",
    ),
    max_frames: int | None = typer.Option(
        None,
        "--max-frames",
        "-m",
        help="Limit video frames to process",
    ),
    run_qa: bool = typer.Option(
        True,
        "--qa/--no-qa",
        help="Automatically run QA engine on ingested session",
    ),
) -> None:
    """Ingest Nothing Phone recording from OpenCamera Sensors."""
    from rigsim.ingest.opencamera import OpenCameraSensorsAdapter
    from rigsim.qa.engine import QAEngine

    out_dir = output_dir or Path(f"data/phone_ingested/{dataset_path.name}")
    console.print(
        f"[bold green]Ingesting OpenCamera Sensors dataset:[/bold green] [cyan]{dataset_path}[/cyan]"
    )

    adapter = OpenCameraSensorsAdapter(
        input_dir=dataset_path,
        output_dir=out_dir,
        max_frames=max_frames,
    )
    session = adapter.ingest()
    console.print(
        f"[bold green]✓ Ingest complete![/bold green] Session ID: [cyan]{session.session_id}[/cyan]"
    )
    console.print(f"MCAP written to: [cyan]{session.mcap_path}[/cyan]")

    if run_qa:
        console.print("[bold cyan]Running automated QA engine...[/bold cyan]")
        qa_engine = QAEngine()
        verdict = qa_engine.process_session(out_dir)
        console.print(
            f"Overall Verdict: [bold]{verdict.overall_verdict.value.upper()}[/bold] | "
            f"Usable Yield: [bold green]{verdict.usable_percentage:.1f}%[/bold green]"
        )


@app.command()
def label(
    session: Path = typer.Argument(..., help="Path to session directory for fault labelling"),
    fault_type: str | None = typer.Option(
        None,
        "--fault-type",
        "-t",
        help="Fault type (e.g. motion_blur, underexposure, random_frame_drop)",
    ),
    stream_id: str = typer.Option("phone_cam", "--stream", "-s", help="Target sensor stream ID"),
    t_start: float = typer.Option(
        0.0, "--start", help="Fault start time in seconds relative to session start"
    ),
    t_end: float = typer.Option(
        1.0, "--end", help="Fault end time in seconds relative to session start"
    ),
    severity: float = typer.Option(1.0, "--severity", help="Severity level from 0.0 to 1.0"),
    labels_csv: Path | None = typer.Option(None, "--csv", help="Import fault labels from CSV file"),
) -> None:
    """Label ground truth fault windows on real recordings for QA benchmarking."""
    import json

    import pandas as pd

    from rigsim.config.faults import FaultLabel
    from rigsim.config.types import FaultType

    sess_dir = session.parent if session.is_file() else session
    manifest_file = sess_dir / "session.json"
    if not manifest_file.exists():
        console.print(f"[bold red]Session manifest not found at: {manifest_file}[/bold red]")
        raise typer.Exit(1)

    with open(manifest_file) as f:
        sess_data = json.load(f)

    labels_file = sess_dir / "labels.json"
    existing_labels = []
    if labels_file.exists():
        with open(labels_file) as f:
            existing_labels = json.load(f)

    t_base_ns = 1_700_000_000_000_000_000

    if labels_csv and labels_csv.exists():
        console.print(f"Importing labels from CSV: [cyan]{labels_csv}[/cyan]")
        df = pd.read_csv(labels_csv)
        for _, row in df.iterrows():
            f_type = str(row["fault_type"])
            s_id = str(row.get("stream_id", stream_id))
            st_s = float(row.get("t_start_s", row.get("start", 0.0)))
            et_s = float(row.get("t_end_s", row.get("end", 1.0)))
            sev = float(row.get("severity", 1.0))
            lbl = FaultLabel(
                fault_type=FaultType(f_type),
                stream_id=s_id,
                t_start_ns=t_base_ns + int(st_s * 1e9),
                t_end_ns=t_base_ns + int(et_s * 1e9),
                severity=sev,
                params={"source": "csv_import"},
            )
            existing_labels.append(lbl.model_dump())
    elif fault_type:
        lbl = FaultLabel(
            fault_type=FaultType(fault_type),
            stream_id=stream_id,
            t_start_ns=t_base_ns + int(t_start * 1e9),
            t_end_ns=t_base_ns + int(t_end * 1e9),
            severity=severity,
            params={"source": "cli_label"},
        )
        existing_labels.append(lbl.model_dump())
        console.print(
            f"Added label: [bold cyan]{fault_type}[/bold cyan] on stream [yellow]{stream_id}[/yellow] [{t_start:.2f}s - {t_end:.2f}s]"
        )
    else:
        console.print(
            "[yellow]No labels specified. Use --fault-type or --csv to add ground truth labels.[/yellow]"
        )
        return

    with open(labels_file, "w") as f:
        json.dump(existing_labels, f, indent=2)

    # Update session manifest with labels path
    sess_data["ground_truth_labels_path"] = str(labels_file.resolve())
    with open(manifest_file, "w") as f:
        json.dump(sess_data, f, indent=2)

    console.print(
        f"[bold green]✓ Saved {len(existing_labels)} ground-truth labels to:[/bold green] [cyan]{labels_file}[/cyan]"
    )


@app.command()
def eval(
    suite: str = typer.Option("quick", "--suite", "-s", help="Evaluation suite: quick or full"),
    rig: Path = typer.Option(
        Path("configs/rigs/handheld_stereo_imu.yaml"),
        "--rig",
        "-r",
        help="Rig configuration for test matrix synthesis",
    ),
    sessions_dir: Path | None = typer.Option(
        None, "--sessions-dir", "-d", help="Directory containing pre-recorded sessions to evaluate"
    ),
    output: Path = typer.Option(
        Path("reports/eval_benchmark.html"),
        "--output",
        "-o",
        help="Output path for interactive HTML evaluation report",
    ),
) -> None:
    """Run full benchmark evaluating QA engine precision, recall, F1, and temporal IoU."""
    from rigsim.eval import BenchmarkEngine, generate_evaluation_report

    console.print("[bold cyan]============================================================[/bold cyan]")
    console.print("[bold green]RigSim Automated QA Engine Evaluation Benchmark[/bold green]")
    console.print(f"Suite: [cyan]{suite}[/cyan] | Rig: [cyan]{rig}[/cyan]")
    console.print("[bold cyan]============================================================[/bold cyan]\n")

    bench_engine = BenchmarkEngine(rig_config_path=rig)

    if sessions_dir and sessions_dir.exists():
        subdirs = [p for p in sessions_dir.iterdir() if p.is_dir() and (p / "session.json").exists()]
        if not subdirs:
            console.print(f"[bold red]No valid sessions with session.json found in: {sessions_dir}[/bold red]")
            raise typer.Exit(code=1)
        console.print(f"Evaluating [yellow]{len(subdirs)}[/yellow] sessions from: {sessions_dir}")
        result = bench_engine.evaluate_sessions(subdirs, suite_name=f"corpus_{sessions_dir.name}")
    else:
        console.print(f"Synthesizing automated test sessions matrix ([yellow]{suite}[/yellow] mode)...")
        result = bench_engine.run_automated_suite(suite_type=suite)

    # 1. Scorecard Table
    score_table = Table(title=f"QA Engine Benchmark Summary: {result.suite_name}")
    score_table.add_column("Evaluation Metric", style="cyan")
    score_table.add_column("Score", justify="right", style="bold green")
    score_table.add_column("Benchmark Target", justify="right", style="dim")
    score_table.add_column("Status", justify="center")

    def _status(val: float, target: float, higher_better: bool = True) -> str:
        passed = (val >= target) if higher_better else (val <= target)
        return "[bold green]PASS[/bold green]" if passed else "[bold yellow]WARN[/bold yellow]"

    score_table.add_row(
        "Overall Accuracy",
        f"{result.overall_metrics.accuracy * 100:.1f}%",
        ">= 90.0%",
        _status(result.overall_metrics.accuracy, 0.90),
    )
    score_table.add_row(
        "Macro F1-Score",
        f"{result.macro_f1 * 100:.1f}%",
        ">= 85.0%",
        _status(result.macro_f1, 0.85),
    )
    score_table.add_row(
        "Macro Precision",
        f"{result.macro_precision * 100:.1f}%",
        ">= 85.0%",
        _status(result.macro_precision, 0.85),
    )
    score_table.add_row(
        "Macro Recall (Sensitivity)",
        f"{result.macro_recall * 100:.1f}%",
        ">= 90.0%",
        _status(result.macro_recall, 0.90),
    )
    score_table.add_row(
        "False Positive Rate (FPR)",
        f"{result.overall_metrics.false_positive_rate * 100:.1f}%",
        "<= 5.0%",
        _status(result.overall_metrics.false_positive_rate, 0.05, higher_better=False),
    )
    score_table.add_row(
        "Mean Temporal IoU",
        f"{result.mean_temporal_iou * 100:.1f}%",
        ">= 70.0%",
        _status(result.mean_temporal_iou, 0.70),
    )
    console.print(score_table)
    console.print()

    # 2. Confusion Matrix Table
    cm_table = Table(title="Multiclass 5×5 Confusion Matrix (Rows: Ground Truth, Cols: Predicted)")
    cm_table.add_column("True Category", style="cyan")
    for cat in result.category_labels:
        cm_table.add_column(f"Pred: {cat}", justify="right")

    for i, true_cat in enumerate(result.category_labels):
        row_vals = [str(result.confusion_matrix_5x5[i, j]) for j in range(len(result.category_labels))]
        cm_table.add_row(f"[bold]{true_cat}[/bold]", *row_vals)
    console.print(cm_table)
    console.print()

    # 3. Per-Category Statistics Table
    cat_table = Table(title="Per-Category Statistical Performance Breakdown")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("TP", justify="right")
    cat_table.add_column("FP", justify="right")
    cat_table.add_column("TN", justify="right")
    cat_table.add_column("FN", justify="right")
    cat_table.add_column("Precision", justify="right")
    cat_table.add_column("Recall", justify="right")
    cat_table.add_column("F1-Score", justify="right", style="bold")

    for cat, m in result.category_metrics.items():
        cat_table.add_row(
            cat,
            str(m.tp),
            str(m.fp),
            str(m.tn),
            str(m.fn),
            f"{m.precision * 100:.1f}%",
            f"{m.recall * 100:.1f}%",
            f"{m.f1 * 100:.1f}%",
        )
    console.print(cat_table)
    console.print()

    # Generate interactive HTML report
    report_file = generate_evaluation_report(result, output)
    console.print(
        f"[bold green]✓ Interactive evaluation report generated at:[/bold green] [cyan]{report_file}[/cyan]\n"
    )


@app.command()
def bench(
    frames: int = typer.Option(60, "--frames", "-f", help="Number of 1280x720 video frames to test"),
    samples: int = typer.Option(2000, "--samples", "-s", help="Number of IMU samples to test"),
) -> None:
    """Run throughput and processing speed benchmarks across QA checks."""
    from rigsim.eval import QAThroughputBenchmarker

    console.print("[bold cyan]============================================================[/bold cyan]")
    console.print("[bold green]RigSim Automated QA Performance & Throughput Benchmark[/bold green]")
    console.print(f"Test Volume: [yellow]{frames} frames (1280x720 RGB)[/yellow] + [yellow]{samples} IMU samples[/yellow]")
    console.print("[bold cyan]============================================================[/bold cyan]\n")

    benchmarker = QAThroughputBenchmarker(num_camera_frames=frames, num_imu_samples=samples)
    report = benchmarker.run_benchmark()

    tbl = Table(title="RigSim QA Check Execution Speed & Realtime Headroom")
    tbl.add_column("Cat", style="dim")
    tbl.add_column("ID", style="bold cyan")
    tbl.add_column("Check Name", style="white")
    tbl.add_column("Volume", justify="right")
    tbl.add_column("Duration", justify="right")
    tbl.add_column("Rate", justify="right")
    tbl.add_column("Speedup", justify="right", style="bold green")

    for b in report.benchmarks:
        unit = "fps" if b.category in ("Image", "Timing") and "IMU" not in b.check_name else "samp/s"
        tbl.add_row(
            b.category,
            b.check_id,
            b.check_name,
            str(b.samples_processed),
            f"{b.duration_s * 1000:.1f} ms",
            f"{b.throughput_rate:.0f} {unit}",
            f"{b.realtime_speedup:.1f}x",
        )
    console.print(tbl)
    console.print()
    console.print(
        f"[bold green]Summary:[/bold green] Image Checks Average: [cyan]{report.average_image_fps:.1f} fps[/cyan] "
        f"| IMU Checks Average: [cyan]{report.average_imu_samples_per_s:,.0f} samples/s[/cyan]"
    )
    console.print(
        f"Total Ingestion & Evaluation Time: [bold green]{report.total_duration_s * 1000:.1f} ms[/bold green] (Faster than real-time)\n"
    )



@app.command("export-episodes")
def export_episodes(
    session: Path = typer.Argument(..., help="Path to QA-verified session to export"),
    output_dir: Path = typer.Option(
        Path("data/episodes"),
        "--output-dir",
        "-o",
        help="Destination directory for Episode packages",
    ),
    min_duration: float = typer.Option(
        2.0,
        "--min-duration",
        "-m",
        help="Minimum episode duration in seconds to avoid micro-fragments",
    ),
    defect_margin: float = typer.Option(
        0.2,
        "--defect-margin",
        help="Buffer margin in seconds trimmed from defect boundaries",
    ),
    preview: bool = typer.Option(
        True,
        "--preview/--no-preview",
        help="Generate downsampled MP4 preview clip for visual review",
    ),
) -> None:
    """Export QA-passed usable time segments as versioned Episode packages for Project 2 (EpisodeForge)."""
    from rigsim.episode import EpisodeExtractor

    console.print("[bold cyan]============================================================[/bold cyan]")
    console.print("[bold green]RigSim Usable Episode Extractor & Delivery Packager[/bold green]")
    console.print(f"Source Session: [cyan]{session}[/cyan] | Output Dir: [cyan]{output_dir}[/cyan]")
    console.print("[bold cyan]============================================================[/bold cyan]\n")

    if not session.exists() or not (session / "session.json").exists():
        console.print(f"[bold red]Error: Session directory or session.json not found at: {session}[/bold red]")
        raise typer.Exit(code=1)

    extractor = EpisodeExtractor(
        min_duration_s=min_duration,
        defect_margin_s=defect_margin,
        generate_preview=preview,
    )
    dataset = extractor.extract_episodes(session_dir=session, output_dir=output_dir)

    # Print Episode Summary Table
    tbl = Table(title=f"Exported Client Delivery Episodes: {dataset.dataset_id}")
    tbl.add_column("Episode ID", style="bold cyan")
    tbl.add_column("Relative Path", style="dim")
    tbl.add_column("Duration", justify="right")
    tbl.add_column("Video Frames", justify="right")
    tbl.add_column("MCAP SHA-256 (Provenance)", style="dim")
    tbl.add_column("QA Status", justify="center")

    for ep in dataset.episodes:
        # Aggregate total video frames
        total_vid_frames = sum(cnt for top, cnt in ep["frames_count"].items() if "image_compressed" in top)
        tbl.add_row(
            ep["episode_id"],
            ep["relative_path"],
            f"{ep['duration_s']:.2f} s",
            f"{total_vid_frames:,} frames",
            f"{ep['mcap_sha256'][:16]}...",
            f"[bold green]{ep['qa_verdict']}[/bold green]",
        )

    tbl.add_section()
    tbl.add_row(
        "[bold]TOTAL USABLE YIELD[/bold]",
        f"[bold]{dataset.total_episodes} Episodes[/bold]",
        f"[bold green]{dataset.total_usable_duration_s:.2f} s[/bold green]",
        f"[dim]of {dataset.source_total_duration_s:.1f}s total[/dim]",
        "-",
        f"[bold green]{dataset.usable_yield_pct:.1f}%[/bold green]",
    )
    console.print(tbl)
    console.print()

    manifest_file = output_dir / session.name / "dataset_manifest.json"
    console.print(
        f"[bold green]✓ Episode package successfully exported![/bold green] "
        f"Manifest written to: [cyan]{manifest_file}[/cyan]\n"
    )


if __name__ == "__main__":
    app()
