#!/usr/bin/env python3
"""Fullscreen High-Refresh-Rate QR Clock generator for cross-device optical clock synchronization."""

from pathlib import Path
import time

import cv2
import numpy as np
import qrcode
import typer
from rich.console import Console

app = typer.Typer(help="High-refresh QR Clock for cross-device camera-screen synchronization.")
console = Console()


@app.command()
def run(
    duration_s: float = typer.Option(60.0, "--duration", "-d", help="Run duration in seconds (0 for indefinite)"),
    log_csv: Path = typer.Option(Path("data/qr_clock_log.csv"), "--log", "-l", help="Path to write display timestamps CSV"),
    fullscreen: bool = typer.Option(True, "--fullscreen/--windowed", help="Display fullscreen"),
    qr_size_px: int = typer.Option(600, "--size", "-s", help="QR code display width/height in pixels"),
) -> None:
    """Run optical QR Clock encoding Mac time.time_ns() on screen at display refresh rate."""
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    f_log = open(log_csv, "w")
    f_log.write("frame_count,display_time_ns,dt_ms\n")

    window_name = "RigSim Precision Optical QR Clock"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    console.print(f"[bold green]Starting QR Clock[/bold green] (Duration: {duration_s}s, Fullscreen: {fullscreen})")
    console.print("Point your phone camera steadily at the screen. Press [bold cyan]'q'[/bold cyan] or [bold cyan]ESC[/bold cyan] to exit.")

    frame_count = 0
    t_start = time.time()
    t_end = t_start + duration_s if duration_s > 0 else float("inf")
    last_t_ns = time.time_ns()

    screen_w, screen_h = 1920, 1080

    try:
        while time.time() < t_end:
            now_ns = time.time_ns()
            dt_ms = (now_ns - last_t_ns) / 1e6
            last_t_ns = now_ns

            # 1. Generate QR Code
            payload = f'{{"t_ns": {now_ns}, "frame": {frame_count}}}'
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=10,
                border=2,
            )
            qr.add_data(payload)
            qr.make(fit=True)

            mat = np.array(qr.get_matrix(), dtype=np.uint8)
            qr_raw = np.where(mat, 0, 255).astype(np.uint8)
            qr_display = cv2.resize(qr_raw, (qr_size_px, qr_size_px), interpolation=cv2.INTER_NEAREST)

            # 2. Build full canvas
            canvas = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 245

            # Center QR code
            x_offset = (screen_w - qr_size_px) // 2
            y_offset = (screen_h - qr_size_px) // 2
            canvas[y_offset : y_offset + qr_size_px, x_offset : x_offset + qr_size_px] = cv2.cvtColor(
                qr_display, cv2.COLOR_GRAY2BGR
            )

            # 3. Add Human-Readable Header and Timestamps
            dt_readable = datetime_str = time.strftime("%H:%M:%S", time.localtime(now_ns / 1e9))
            millis = int((now_ns % 1_000_000_000) / 1_000_000)
            time_display = f"{dt_readable}.{millis:03d} UTC"

            cv2.putText(
                canvas,
                "RigSim Optical Synchronization Clock",
                (screen_w // 2 - 380, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                time_display,
                (screen_w // 2 - 250, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 120, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Frame #{frame_count:06d}  |  Unix: {now_ns} ns",
                (screen_w // 2 - 320, screen_h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (80, 80, 80),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, canvas)
            f_log.write(f"{frame_count},{now_ns},{dt_ms:.3f}\n")
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        f_log.close()
        cv2.destroyAllWindows()

    total_elapsed = time.time() - t_start
    avg_fps = frame_count / max(0.01, total_elapsed)
    console.print(f"\n[bold green]✓ QR Clock finished![/bold green] Displayed {frame_count} frames in {total_elapsed:.1f}s")
    console.print(f"Average display refresh rate: [bold cyan]{avg_fps:.1f} Hz[/bold cyan]")
    console.print(f"Display event log saved to: [cyan]{log_csv}[/cyan]")


if __name__ == "__main__":
    app()
