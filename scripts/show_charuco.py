#!/usr/bin/env python3
"""Fullscreen ChArUco calibration board display with physical screen DPI millimeter scaling."""

import cv2
import numpy as np
import typer
from rich.console import Console

app = typer.Typer(help="Display full-screen ChArUco calibration target.")
console = Console()


@app.command()
def show(
    squares_x: int = typer.Option(7, "--squares-x", "-x", help="Number of chessboard squares horizontally"),
    squares_y: int = typer.Option(5, "--squares-y", "-y", help="Number of chessboard squares vertically"),
    square_len_mm: float = typer.Option(40.0, "--square-mm", "-s", help="Physical square size in millimeters"),
    marker_len_mm: float = typer.Option(30.0, "--marker-mm", "-m", help="Physical ArUco marker size in millimeters"),
    dpi: float = typer.Option(254.0, "--dpi", "-d", help="Screen display DPI (Retina MacBook Air/Pro is ~254 DPI)"),
    fullscreen: bool = typer.Option(True, "--fullscreen/--windowed", help="Display full screen"),
) -> None:
    """Display ChArUco calibration board full screen calibrated to screen DPI."""
    mm_per_pixel = 25.4 / dpi
    square_px = int(round(square_len_mm / mm_per_pixel))
    marker_px = int(round(marker_len_mm / mm_per_pixel))

    console.print(f"[bold green]Generating ChArUco Board:[/bold green] {squares_x}x{squares_y} grid")
    console.print(f"Target Square: [cyan]{square_len_mm} mm[/cyan] ({square_px} px at {dpi:.1f} DPI)")
    console.print(f"Target Marker: [cyan]{marker_len_mm} mm[/cyan] ({marker_px} px)")
    console.print(
        "[bold yellow]Operator Calibration Verification:[/bold yellow]\n"
        "Place a physical ruler against the screen to verify that one square is exactly 40.0 mm wide."
    )

    board_w = squares_x * square_px
    board_h = squares_y * square_px

    # Create ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len_mm / 1000.0, marker_len_mm / 1000.0, aruco_dict)

    board_img = board.generateImage((board_w, board_h), marginSize=int(square_px * 0.5), borderBits=1)

    # Fullscreen canvas
    screen_w, screen_h = 1920, 1080
    canvas = np.ones((screen_h, screen_w), dtype=np.uint8) * 255

    # Center board on canvas
    x_off = max(0, (screen_w - board_w) // 2)
    y_off = max(0, (screen_h - board_h) // 2)
    paste_w = min(board_w, screen_w - x_off)
    paste_h = min(board_h, screen_h - y_off)

    canvas[y_off : y_off + paste_h, x_off : x_off + paste_w] = board_img[:paste_h, :paste_w]

    window_name = "RigSim ChArUco Calibration Target"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    console.print("Press [bold cyan]'q'[/bold cyan] or [bold cyan]ESC[/bold cyan] to close window.")
    while True:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()

