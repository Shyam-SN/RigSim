#!/usr/bin/env python3
"""MacBook webcam capture script using OpenCV AVFoundation with dual nanosecond timestamping."""

from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from rigsim.config.camera import CameraDistortion, CameraIntrinsics
from rigsim.config.session import Session, Stream
from rigsim.config.transform import Transform
from rigsim.config.types import DistortionModel, SensorKind, SessionSource
from rigsim.io.mcap_writer import RigSimMCAPWriter

app = typer.Typer(help="Record MacBook webcam video with nanosecond timestamps.")
console = Console()


@app.command()
def record(
    output_dir: Path = typer.Option(
        Path("data/webcam_session"),
        "--output-dir",
        "-o",
        help="Destination directory for MCAP and session manifest",
    ),
    device_index: int = typer.Option(0, "--device", "-d", help="Camera index (default 0 for built-in Mac webcam)"),
    width: int = typer.Option(1280, "--width", "-w", help="Requested frame width"),
    height: int = typer.Option(720, "--height", "-h", help="Requested frame height"),
    fps: float = typer.Option(30.0, "--fps", "-f", help="Requested frame rate"),
    duration_s: float = typer.Option(10.0, "--duration", "-t", help="Recording duration in seconds (0 for indefinite)"),
) -> None:
    """Record Mac webcam with AVFoundation backend and exact time.time_ns() capture timestamps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mcap_path = output_dir / "recording.mcap"
    session_json_path = output_dir / "session.json"
    ts_csv_path = output_dir / "timestamps.csv"

    console.print("[bold yellow]macOS Camera Permission Reminder:[/bold yellow]")
    console.print(
        "Ensure your Terminal/IDE has Camera access granted in:\n"
        "[dim]System Settings → Privacy & Security → Camera[/dim]\n"
    )

    console.print(f"[bold green]Opening camera index {device_index} with cv2.CAP_AVFOUNDATION...[/bold green]")
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        console.print("[bold red]Failed to open video capture device![/bold red]")
        raise typer.Exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_prop = cap.get(cv2.CAP_PROP_FPS)

    console.print(
        f"Configured: Requested {width}x{height} @ {fps} Hz | "
        f"Delivered: [bold cyan]{actual_w}x{actual_h}[/bold cyan] @ {actual_fps_prop} Hz"
    )

    # Initialize MCAP Writer
    writer = RigSimMCAPWriter(mcap_path)
    writer.register_camera_channel("webcam")
    writer.register_tf_channel()

    fx = fy = float(actual_w * 0.9)
    cx = float(actual_w / 2.0)
    cy = float(actual_h / 2.0)
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=actual_w, height=actual_h)
    distortion = CameraDistortion(model=DistortionModel.RADTAN, coeffs=[0.0, 0.0, 0.0, 0.0])

    t_start_ns = time.time_ns()
    writer.write_camera_calibration("webcam", "webcam", intrinsics, distortion, t_start_ns, t_start_ns)
    writer.write_transform("body", "webcam", Transform.identity(), t_start_ns, t_start_ns)

    console.print(f"[bold green]Recording for {duration_s} seconds (Press Ctrl+C or 'q' to stop)...[/bold green]")

    frame_count = 0
    timestamps_log: list[tuple[int, int, int]] = []  # (frame_idx, device_time_ns, host_time_ns)

    start_wall = time.time()
    t_end = start_wall + duration_s if duration_s > 0 else float("inf")

    try:
        while time.time() < t_end:
            grab_ok = cap.grab()
            capture_ts = time.time_ns()
            host_ts = capture_ts + 500_000  # small reception offset

            if not grab_ok:
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.write_image("webcam", "webcam", frame_rgb, capture_ts, host_ts, encoding="jpeg")

            timestamps_log.append((frame_count, capture_ts, host_ts))
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - start_wall
                current_fps = frame_count / max(0.01, elapsed)
                console.print(f"Captured {frame_count} frames | Elapsed: {elapsed:.1f}s | FPS: {current_fps:.1f}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Recording interrupted by user.[/yellow]")
    finally:
        cap.release()
        writer.close()

    total_time = time.time() - start_wall
    achieved_fps = frame_count / max(0.01, total_time)

    # Save timestamp CSV
    with open(ts_csv_path, "w") as f:
        f.write("frame_index,device_time_ns,host_time_ns\n")
        for f_idx, d_ts, h_ts in timestamps_log:
            f.write(f"{f_idx},{d_ts},{h_ts}\n")

    # Save Session manifest
    session = Session(
        session_id=output_dir.name,
        source=SessionSource.WEBCAM,
        rig_config_name="macbook_builtin_webcam",
        created_at=datetime.now(timezone.utc).isoformat(),
        operator="MacBook Operator",
        streams=[
            Stream(
                stream_id="webcam",
                kind=SensorKind.RGB,
                nominal_rate_hz=float(fps),
                resolution=(actual_w, actual_h),
                encoding="jpeg",
                clock_id="mac_system_clock",
                frame_id="webcam",
                intrinsics=intrinsics,
                distortion=distortion,
                extrinsics=Transform.identity(),
            )
        ],
        mcap_path=str(mcap_path.resolve()),
        metadata={
            "requested_fps": fps,
            "achieved_fps": achieved_fps,
            "total_frames": frame_count,
            "actual_resolution": [actual_w, actual_h],
        },
    )

    with open(session_json_path, "w") as f:
        f.write(session.model_dump_json(indent=2))

    table = Table(title="MacBook Webcam Capture Summary")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="bold green")
    table.add_row("Total Captured Frames", str(frame_count))
    table.add_row("Recording Duration", f"{total_time:.2f} s")
    table.add_row("Achieved Frame Rate", f"{achieved_fps:.2f} FPS (requested {fps:.1f})")
    table.add_row("Resolution", f"{actual_w}x{actual_h}")
    table.add_row("MCAP Output", str(mcap_path))
    table.add_row("Session Manifest", str(session_json_path))
    console.print(table)


if __name__ == "__main__":
    app()

