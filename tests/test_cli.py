"""Tests for RigSim CLI."""

from typer.testing import CliRunner

from rigsim.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Multi-Camera + IMU Capture Simulator" in result.stdout
    assert "simulate" in result.stdout
    assert "qa" in result.stdout
    assert "report" in result.stdout
    assert "plan" in result.stdout
    assert "calibrate" in result.stdout
    assert "sync" in result.stdout


def test_cli_simulate_help():
    result = runner.invoke(app, ["simulate", "--help"])
    assert result.exit_code == 0
    assert "--rig" in result.stdout
    assert "--faults" in result.stdout
    assert "--duration" in result.stdout
    assert "--seed" in result.stdout


def test_cli_sync_help():
    result = runner.invoke(app, ["sync", "--help"])
    assert result.exit_code == 0
    assert "qr" in result.stdout
    assert "imu-cam" in result.stdout
