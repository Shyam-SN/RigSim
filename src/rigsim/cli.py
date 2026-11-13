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
        None, "--bench-clips", help="Sample video clips for empirical codec benchmarking"
    ),
) -> None:
    """Run capture planner: calculate bandwidth, test codecs, model storage and cloud costs."""
    console.print(f"[bold green]Running capture planner on:[/bold green] [cyan]{rig}[/cyan]")


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
    phone_session: Path = typer.Argument(..., help="Path to phone capture session"),
    mac_log: Path = typer.Option(
        ..., "--mac-log", "-m", help="Path to Mac QR clock timestamp CSV log"
    ),
) -> None:
    """Perform cross-device QR clock synchronization and fit clock model with robust regression."""
    console.print(f"[bold green]Syncing phone session:[/bold green] [cyan]{phone_session}[/cyan]")


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


@app.command()
def label(
    session: Path = typer.Argument(
        ..., help="Path to session for manual/interactive fault labelling"
    ),
) -> None:
    """Interactive tool to label ground truth fault windows on real recordings."""
    console.print(f"[bold green]Labelling session:[/bold green] [cyan]{session}[/cyan]")


@app.command()
def eval(
    suite: str = typer.Option("quick", "--suite", "-s", help="Evaluation suite: quick or full"),
) -> None:
    """Run full benchmark evaluating QA engine precision, recall, F1, and temporal IoU."""
    console.print(
        f"[bold green]Running QA Engine evaluation suite:[/bold green] [cyan]{suite}[/cyan]"
    )


@app.command()
def bench() -> None:
    """Run throughput and processing speed benchmarks across QA checks."""
    console.print("[bold green]Running QA throughput benchmarks...[/bold green]")


@app.command("export-episodes")
def export_episodes(
    session: Path = typer.Argument(..., help="Path to QA-verified session to export"),
    output_dir: Path = typer.Option(
        Path("data/episodes"),
        "--output-dir",
        "-o",
        help="Destination directory for Episode packages",
    ),
) -> None:
    """Export QA-passed usable time segments as versioned Episode packages for Project 2 (EpisodeForge)."""
    console.print(
        f"[bold green]Exporting usable episodes from:[/bold green] [cyan]{session}[/cyan]"
    )


if __name__ == "__main__":
    app()
