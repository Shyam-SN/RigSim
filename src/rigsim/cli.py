"""RigSim Command Line Interface."""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="rigsim",
    help="Multi-Camera + IMU Capture Simulator, Real Rig, and Automated Data QA Engine",
    add_completion=False,
)
sync_app = typer.Typer(name="sync", help="Sensor synchronization tools (QR clock, IMU-camera)")
app.add_typer(sync_app, name="sync")

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
    # Implementation wired in Milestone M1
    console.print(
        "[yellow]M0 CLI skeleton: Milestone M1 will generate the full MCAP session.[/yellow]"
    )


@app.command()
def calibrate(
    session_or_folder: Path = typer.Argument(
        ..., help="Path to session directory or image folder with ChArUco targets"
    ),
) -> None:
    """Perform intrinsic and stereo calibration using ChArUco fiducials."""
    console.print(
        f"[bold green]Running calibration on:[/bold green] [cyan]{session_or_folder}[/cyan]"
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


@app.command()
def report(
    session: Path | None = typer.Argument(
        None, help="Path to session.json for single-session report"
    ),
    batch: Path | None = typer.Option(
        None, "--batch", "-b", help="Folder containing multiple sessions for multi-session summary"
    ),
) -> None:
    """Generate offline self-contained HTML report (single session or multi-session batch)."""
    if batch:
        console.print(
            f"[bold green]Generating multi-session summary report for:[/bold green] [cyan]{batch}[/cyan]"
        )
    elif session:
        console.print(
            f"[bold green]Generating single-session HTML report for:[/bold green] [cyan]{session}[/cyan]"
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
) -> None:
    """Estimate IMU-camera temporal offset via visual angular velocity and gyroscope cross-correlation."""
    console.print(
        f"[bold green]Estimating IMU-camera offset for:[/bold green] [cyan]{session}[/cyan]"
    )


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
