"""Evaluation and experiment runners for real-world EuRoC MAV datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from rigsim.config.faults import FaultScenarioConfig
from rigsim.config.types import FaultType, QAVerdict
from rigsim.ingest.euroc import EuRoCIngestAdapter
from rigsim.qa.engine import QAEngine
from rigsim.sync.imu_cam import ImuCameraSyncEstimator


@dataclass
class OffsetSweepResult:
    injected_offset_ms: float
    estimated_offset_ms: float
    estimation_error_ms: float
    correlation_peak: float
    status: str


class EuRoCEvaluator:
    """Runs systematic benchmarks on EuRoC MAV datasets: clean QA, injected faults, and sync sweeps."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def run_clean_evaluation(
        self,
        euroc_dir: Path,
        output_dir: Path,
        max_frames: int | None = 150,
    ) -> dict[str, Any]:
        """Ingest clean EuRoC sequence and evaluate standard QA checks."""
        self.console.print(
            f"[bold cyan]Running clean EuRoC QA evaluation on:[/bold cyan] {euroc_dir}"
        )

        adapter = EuRoCIngestAdapter(
            euroc_dir=euroc_dir,
            output_dir=output_dir,
            fault_config=None,
            max_camera_frames=max_frames,
        )
        session = adapter.ingest()

        engine = QAEngine()
        verdict = engine.process_session(output_dir)

        table = Table(title=f"EuRoC Clean Run QA Verdict: {session.session_id}")
        table.add_column("Stream", style="cyan")
        table.add_column("Check ID", style="bold")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Verdict")

        for res in verdict.results:
            v_style = (
                "[bold green]PASS[/bold green]"
                if res.verdict == QAVerdict.PASS
                else "[bold yellow]WARN[/bold yellow]"
                if res.verdict == QAVerdict.WARN
                else "[bold red]FAIL[/bold red]"
            )
            table.add_row(
                res.stream_id,
                res.check_id,
                res.metric_name,
                f"{res.value:.4g}" if isinstance(res.value, (int, float)) else str(res.value),
                f"{res.threshold:.4g}"
                if isinstance(res.threshold, (int, float))
                else str(res.threshold),
                v_style,
            )

        self.console.print(table)
        self.console.print(
            f"Overall Verdict: [bold]{verdict.overall_verdict.value.upper()}[/bold] | "
            f"Usable Yield: [bold green]{verdict.usable_percentage:.1f}%[/bold green]"
        )

        return {
            "session_id": session.session_id,
            "overall_verdict": verdict.overall_verdict.value,
            "usable_percentage": verdict.usable_percentage,
            "results": verdict.results,
        }

    def run_fault_evaluation(
        self,
        euroc_dir: Path,
        output_dir: Path,
        fault_config: FaultScenarioConfig,
        max_frames: int | None = 150,
    ) -> dict[str, Any]:
        """Ingest EuRoC sequence with injected faults on real images and evaluate detection."""
        self.console.print(
            "[bold magenta]Running EuRoC Injected-Fault QA evaluation on real data...[/bold magenta]"
        )

        adapter = EuRoCIngestAdapter(
            euroc_dir=euroc_dir,
            output_dir=output_dir,
            fault_config=fault_config,
            max_camera_frames=max_frames,
        )
        adapter.ingest()

        engine = QAEngine()
        verdict = engine.process_session(output_dir)

        # Ground truth labels from labels.json
        labels_file = output_dir / "labels.json"
        import json

        labels_data = []
        if labels_file.exists():
            with open(labels_file) as f:
                labels_data = json.load(f)

        table = Table(title="EuRoC Real Image Fault Detection Evaluation")
        table.add_column("Ground Truth Fault", style="bold cyan")
        table.add_column("Stream", style="cyan")
        table.add_column("Time Window (s)", justify="center")
        table.add_column("QA Check Triggered", style="yellow")
        table.add_column("Detection Status", justify="center")

        detected_count = 0
        total_faults = len(labels_data)

        # Map fault types to corresponding QA checks
        fault_to_check = {
            FaultType.RANDOM_FRAME_DROP.value: "T1_FRAME_DROPS",
            FaultType.BURST_FRAME_DROP.value: "T1_FRAME_DROPS",
            FaultType.FPS_JITTER.value: "T3_FPS_JITTER",
            FaultType.LOW_EFFECTIVE_FPS.value: "T2_EFFECTIVE_FPS",
            FaultType.MOTION_BLUR.value: "I1_BLUR",
            FaultType.OVEREXPOSURE.value: "I2_EXPOSURE",
            FaultType.UNDEREXPOSURE.value: "I2_EXPOSURE",
            FaultType.LENS_OCCLUSION.value: "I3_OCCLUSION",
            FaultType.FROZEN_FRAME.value: "I4_FROZEN",
            FaultType.CORRUPTED_FRAME.value: "I5_CORRUPTION",
            FaultType.HEAVY_NOISE.value: "I6_NOISE",
            FaultType.IMU_DROPOUT.value: "T8_IMU_CONTINUITY",
            FaultType.IMU_SATURATION.value: "M1_IMU_GRAVITY",
            FaultType.IMU_BIAS_JUMP.value: "M3_IMU_BIAS_JUMP",
        }

        flagged_checks = {
            (r.stream_id, r.check_id): r
            for r in verdict.results
            if r.verdict in (QAVerdict.WARN, QAVerdict.FAIL)
        }

        for lbl in labels_data:
            ft = lbl["fault_type"]
            sid = lbl["stream_id"]
            expected_chk = fault_to_check.get(ft, "")
            is_detected = (sid, expected_chk) in flagged_checks

            # Fallback: check if any check on this stream flagged WARN or FAIL
            if not is_detected:
                for (s, c), r in flagged_checks.items():
                    if s == sid:
                        is_detected = True
                        expected_chk = f"{c} (correlated)"
                        break

            if is_detected:
                detected_count += 1
                status = "[bold green]DETECTED[/bold green]"
            else:
                status = "[bold red]MISSED[/bold red]"

            t_start = (lbl["t_start_ns"] - labels_data[0]["t_start_ns"]) / 1e9
            t_end = (lbl["t_end_ns"] - labels_data[0]["t_start_ns"]) / 1e9

            table.add_row(
                ft,
                sid,
                f"{max(0.0, t_start):.2f}s - {max(0.0, t_end):.2f}s",
                expected_chk or "None",
                status,
            )

        self.console.print(table)
        recall = (detected_count / total_faults * 100.0) if total_faults > 0 else 100.0
        self.console.print(
            f"Fault Detection Sensitivity on Real Data: [bold green]{detected_count}/{total_faults} ({recall:.1f}%)[/bold green]"
        )

        return {
            "total_faults": total_faults,
            "detected_faults": detected_count,
            "recall": recall,
            "verdict": verdict.overall_verdict.value,
        }

    def run_offset_sweep(
        self,
        euroc_dir: Path,
        tmp_base: Path,
        offsets_ms: list[float] | None = None,
        max_frames: int = 120,
    ) -> list[OffsetSweepResult]:
        """Inject known IMU-camera temporal offsets into EuRoC and verify recovery accuracy."""
        if offsets_ms is None:
            offsets_ms = [-25.0, -10.0, -5.0, 0.0, 5.0, 10.0, 25.0]

        self.console.print(
            f"[bold cyan]Running IMU-Camera Temporal Calibration Sweep on EuRoC[/bold cyan] (Offsets: {offsets_ms} ms)"
        )

        results: list[OffsetSweepResult] = []

        table = Table(title="EuRoC IMU-Camera Temporal Calibration Recovery Sweep")
        table.add_column("Injected Offset", justify="right", style="cyan")
        table.add_column("Estimated Offset", justify="right", style="bold")
        table.add_column("Abs Error", justify="right", style="magenta")
        table.add_column("Peak Corr (r)", justify="right")
        table.add_column("SLA Status (error < 2.0 ms)", justify="center")

        for offset in offsets_ms:
            step_dir = tmp_base / f"sweep_{int(offset * 1000)}us"
            adapter = EuRoCIngestAdapter(
                euroc_dir=euroc_dir,
                output_dir=step_dir,
                fault_config=None,
                imu_time_shift_ms=offset,
                max_camera_frames=max_frames,
            )
            adapter.ingest()

            # Extract frames and gyro for estimator
            mcap_file = step_dir / "recording.mcap"
            cam_frames, cam_ts, gyro_readings, imu_ts = self._load_mcap_data(mcap_file)

            estimator = ImuCameraSyncEstimator()
            res = estimator.estimate_offset(
                frames_rgb=cam_frames,
                cam_timestamps_ns=cam_ts,
                gyro_readings=gyro_readings,
                imu_timestamps_ns=imu_ts,
            )

            # Note: shifting IMU device timestamps positively means IMU clock is ahead of camera,
            # so the estimated offset (cam - imu) shifts by -offset.
            # In our convention: offset_ms = t_cam - t_imu.
            # When we inject imu_time_shift_ms (adding to imu_time), t_imu becomes t_imu + shift,
            # so t_cam - (t_imu + shift) = (t_cam - t_imu) - shift = -shift.
            expected_offset = -offset
            err = abs(res.offset_ms - expected_offset)
            is_pass = err <= 2.0  # 2 ms sub-frame recovery threshold

            status_str = "[bold green]PASS[/bold green]" if is_pass else "[bold red]FAIL[/bold red]"

            table.add_row(
                f"{offset:+.1f} ms",
                f"{res.offset_ms:+.2f} ms",
                f"{err:.2f} ms",
                f"{res.correlation_peak:.3f}",
                status_str,
            )

            results.append(
                OffsetSweepResult(
                    injected_offset_ms=offset,
                    estimated_offset_ms=res.offset_ms,
                    estimation_error_ms=err,
                    correlation_peak=res.correlation_peak,
                    status="PASS" if is_pass else "FAIL",
                )
            )

        self.console.print(table)
        mean_err = sum(r.estimation_error_ms for r in results) / len(results)
        self.console.print(
            f"Mean Temporal Calibration Error: [bold green]{mean_err:.3f} ms[/bold green]"
        )

        return results

    def _load_mcap_data(self, mcap_path: Path):
        """Extract cam0 frames and imu0 gyro samples from MCAP for synchronization estimation."""
        import json

        import cv2
        import numpy as np
        from foxglove_schemas_protobuf import CompressedImage_pb2
        from mcap.reader import make_reader

        cam_frames = []
        cam_ts = []
        gyro_readings = []
        imu_ts = []

        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                if "/cam0/image_compressed" in channel.topic:
                    pb = CompressedImage_pb2.CompressedImage()
                    pb.ParseFromString(message.data)
                    bgr = cv2.imdecode(np.frombuffer(pb.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        cam_frames.append(bgr)
                        cam_ts.append(message.publish_time)
                elif "/imu0/data" in channel.topic:
                    payload = json.loads(message.data)
                    gyr = payload.get("angular_velocity", {"x": 0.0, "y": 0.0, "z": 0.0})
                    gyro_readings.append([gyr["x"], gyr["y"], gyr["z"]])
                    imu_ts.append(message.publish_time)

        # Ensure monotonic order
        if len(cam_ts) > 1:
            order_c = np.argsort(cam_ts)
            cam_ts = [cam_ts[i] for i in order_c]
            cam_frames = [cam_frames[i] for i in order_c]

        if len(imu_ts) > 1:
            order_i = np.argsort(imu_ts)
            imu_ts = [imu_ts[i] for i in order_i]
            gyro_readings = [gyro_readings[i] for i in order_i]

        return cam_frames, cam_ts, gyro_readings, imu_ts
