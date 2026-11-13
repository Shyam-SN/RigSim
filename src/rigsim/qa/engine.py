"""Automated QA Engine executing all timing and image quality checks."""

from pathlib import Path
import json
import cv2
import numpy as np
import yaml
from foxglove_schemas_protobuf import CompressedImage_pb2
from mcap.reader import make_reader
from rich.console import Console
from rich.table import Table

from rigsim.calib.checker import (
    check_c2_camera_bump,
    check_c3_stereo_consistency,
)
from rigsim.config.qa import QAResult, QAThresholdsConfig, SessionVerdict
from rigsim.config.session import Session
from rigsim.config.types import QAVerdict, SensorKind
from rigsim.qa.image import (
    check_i1_blur,
    check_i2_exposure,
    check_i3_occlusion,
    check_i4_frozen_frames,
    check_i5_corruption,
    check_i6_noise_level,
)
from rigsim.qa.imu import (
    check_m1_imu_gravity,
    check_m2_imu_static_bias,
    check_m3_imu_bias_jump,
    check_m4_imu_gravity_direction,
)
from rigsim.qa.timing import (
    check_t1_frame_drops,
    check_t2_effective_fps,
    check_t3_jitter,
    check_t4_timestamp_sanity,
    check_t5_transport_latency,
    check_t6_inter_camera_sync,
    check_t7_imu_cam_offset,
    check_t8_imu_continuity,
)
from rigsim.qa.verdict import VerdictEvaluator


class QAEngine:
    """End-to-end automated quality assurance engine for multi-sensor capture sessions."""

    def __init__(self, thresholds: QAThresholdsConfig | None = None):
        if thresholds is None:
            with open("configs/qa/default_thresholds.yaml") as f:
                self.thresholds = QAThresholdsConfig.model_validate(yaml.safe_load(f))
        else:
            self.thresholds = thresholds
        self.evaluator = VerdictEvaluator(self.thresholds)
        self.console = Console()

    def process_session(self, session_path: Path) -> SessionVerdict:
        """Run all enabled QA checks on a session and output SessionVerdict."""
        if session_path.is_dir():
            manifest_file = session_path / "session.json"
        else:
            manifest_file = session_path

        with open(manifest_file) as f:
            session = Session.model_validate(json.load(f))

        mcap_path = Path(session.mcap_path)
        if not mcap_path.is_absolute():
            mcap_path = manifest_file.parent / mcap_path.name
        if not mcap_path.exists():
            raise FileNotFoundError(f"MCAP log file not found at: {mcap_path}")

        # 1. Ingest streams from MCAP
        camera_data: dict[str, dict[str, Any]] = {}
        imu_data: dict[str, dict[str, Any]] = {}
        streams_by_id = {s.stream_id: s for s in session.streams}

        for stream in session.streams:
            if stream.kind in [SensorKind.RGB, SensorKind.STEREO_LEFT, SensorKind.STEREO_RIGHT]:
                camera_data[stream.stream_id] = {
                    "nominal_rate_hz": stream.nominal_rate_hz,
                    "device_times": [],
                    "host_times": [],
                    "frames": [],
                    "intrinsics": stream.intrinsics,
                    "distortion": stream.distortion,
                }
            elif stream.kind == SensorKind.IMU:
                imu_data[stream.stream_id] = {
                    "nominal_rate_hz": stream.nominal_rate_hz,
                    "device_times": [],
                    "host_times": [],
                    "is_saturated": [],
                    "accel": [],
                    "gyro": [],
                }

        # Read MCAP sequentially
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                # Camera image message
                if "image_compressed" in channel.topic:
                    # e.g. /cam_stereo_left/image_compressed
                    parts = channel.topic.strip("/").split("/")
                    stream_id = parts[0]
                    if stream_id in camera_data:
                        cam_entry = camera_data[stream_id]
                        cam_entry["device_times"].append(message.publish_time)
                        cam_entry["host_times"].append(message.log_time)

                        # Decode image
                        pb_img = CompressedImage_pb2.CompressedImage()
                        pb_img.ParseFromString(message.data)
                        img_arr = np.frombuffer(pb_img.data, dtype=np.uint8)
                        bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            cam_entry["frames"].append(rgb)

                # IMU sample message
                elif "/data" in channel.topic and "imu" in channel.topic:
                    parts = channel.topic.strip("/").split("/")
                    stream_id = parts[0]
                    if stream_id in imu_data:
                        imu_entry = imu_data[stream_id]
                        imu_entry["device_times"].append(message.publish_time)
                        imu_entry["host_times"].append(message.log_time)
                        payload = json.loads(message.data)
                        imu_entry["is_saturated"].append(payload.get("is_saturated", False))
                        acc = payload.get("linear_acceleration", {"x": 0.0, "y": 0.0, "z": 9.81})
                        gyr = payload.get("angular_velocity", {"x": 0.0, "y": 0.0, "z": 0.0})
                        imu_entry["accel"].append([acc["x"], acc["y"], acc["z"]])
                        imu_entry["gyro"].append([gyr["x"], gyr["y"], gyr["z"]])

        # Sort per-stream samples by device_time_ns to evaluate true sensor physics
        for stream_id, cam_entry in camera_data.items():
            if len(cam_entry["device_times"]) > 1:
                order = np.argsort(cam_entry["device_times"])
                cam_entry["device_times"] = [cam_entry["device_times"][i] for i in order]
                cam_entry["host_times"] = [cam_entry["host_times"][i] for i in order]
                if cam_entry["frames"]:
                    cam_entry["frames"] = [cam_entry["frames"][i] for i in order]

        for stream_id, imu_entry in imu_data.items():
            if len(imu_entry["device_times"]) > 1:
                order = np.argsort(imu_entry["device_times"])
                imu_entry["device_times"] = [imu_entry["device_times"][i] for i in order]
                imu_entry["host_times"] = [imu_entry["host_times"][i] for i in order]
                imu_entry["is_saturated"] = [imu_entry["is_saturated"][i] for i in order]
                imu_entry["accel"] = [imu_entry["accel"][i] for i in order]
                imu_entry["gyro"] = [imu_entry["gyro"][i] for i in order]

        # 2. Run QA Checks
        results: list[QAResult] = []

        all_timestamps = []

        # Camera checks
        for stream_id, cdata in camera_data.items():
            dev_times = cdata["device_times"]
            host_times = cdata["host_times"]
            frames = cdata["frames"]
            rate = cdata["nominal_rate_hz"]

            if dev_times:
                all_timestamps.extend([dev_times[0], dev_times[-1]])

            # Timing checks
            results.append(check_t1_frame_drops(stream_id, dev_times, rate, self.thresholds.timing))
            results.append(check_t2_effective_fps(stream_id, dev_times, rate, self.thresholds.timing))
            results.append(check_t3_jitter(stream_id, dev_times, rate, self.thresholds.timing))
            results.append(check_t4_timestamp_sanity(stream_id, dev_times))
            results.append(check_t5_transport_latency(stream_id, dev_times, host_times, self.thresholds.timing))

            # Image checks
            results.append(check_i1_blur(stream_id, frames, dev_times, self.thresholds.image))
            results.append(check_i2_exposure(stream_id, frames, dev_times, self.thresholds.image))
            results.append(check_i3_occlusion(stream_id, frames, dev_times, self.thresholds.image))
            results.append(check_i4_frozen_frames(stream_id, frames, dev_times, self.thresholds.image))
            results.append(check_i5_corruption(stream_id, frames, dev_times))
            results.append(check_i6_noise_level(stream_id, frames, dev_times))

            # Geometric calibration check (C2: Camera bump via solvePnP)
            if cdata.get("intrinsics") is not None and len(frames) >= 5:
                intr = cdata["intrinsics"]
                K = np.array([
                    [intr.fx, 0, intr.cx],
                    [0, intr.fy, intr.cy],
                    [0, 0, 1.0],
                ], dtype=np.float64)
                dist_coeffs = (
                    np.array(cdata["distortion"].coeffs, dtype=np.float64)
                    if cdata.get("distortion")
                    else np.zeros(5)
                )
                results.append(check_c2_camera_bump(stream_id, frames, dev_times, K, dist_coeffs, self.thresholds.calibration))

        # Inter-camera sync (T6) and Stereo Epipolar Consistency (C3)
        stereo_left_id = None
        stereo_right_id = None
        if "cam_stereo_left" in camera_data and "cam_stereo_right" in camera_data:
            stereo_left_id = "cam_stereo_left"
            stereo_right_id = "cam_stereo_right"
        elif "cam0" in camera_data and "cam1" in camera_data:
            stereo_left_id = "cam0"
            stereo_right_id = "cam1"

        if stereo_left_id and stereo_right_id:
            results.append(check_t6_inter_camera_sync(
                stereo_left_id,
                stereo_right_id,
                camera_data[stereo_left_id]["device_times"],
                camera_data[stereo_right_id]["device_times"],
                self.thresholds.timing,
            ))

            # C3 Stereo Epipolar Consistency
            c_l = camera_data[stereo_left_id]
            c_r = camera_data[stereo_right_id]
            if c_l.get("intrinsics") and c_r.get("intrinsics") and len(c_l["frames"]) > 0 and len(c_r["frames"]) > 0:
                intr_l = c_l["intrinsics"]
                intr_r = c_r["intrinsics"]
                Kl = np.array([[intr_l.fx, 0, intr_l.cx], [0, intr_l.fy, intr_l.cy], [0, 0, 1.0]])
                Kr = np.array([[intr_r.fx, 0, intr_r.cx], [0, intr_r.fy, intr_r.cy], [0, 0, 1.0]])

                s_l = streams_by_id.get(stereo_left_id)
                s_r = streams_by_id.get(stereo_right_id)
                if s_l and s_r:
                    T_l = s_l.extrinsics.as_numpy()
                    T_r = s_r.extrinsics.as_numpy()
                    T_rel = np.linalg.inv(T_r) @ T_l
                    R_rel = T_rel[:3, :3]
                    t_rel = T_rel[:3, 3]
                    tx = np.array([
                        [0, -t_rel[2], t_rel[1]],
                        [t_rel[2], 0, -t_rel[0]],
                        [-t_rel[1], t_rel[0], 0],
                    ])
                    E = tx @ R_rel
                    F = np.linalg.inv(Kr).T @ E @ np.linalg.inv(Kl)
                    results.append(check_c3_stereo_consistency(
                        f"{stereo_left_id}<->{stereo_right_id}",
                        c_l["frames"],
                        c_r["frames"],
                        c_l["device_times"],
                        F,
                        self.thresholds.calibration,
                    ))

        # IMU-Camera Temporal Sync Check (T7)
        primary_cam_id = None
        for cand in ["cam_wrist", "cam_stereo_left", "cam0"]:
            if cand in camera_data and len(camera_data[cand]["frames"]) >= 10:
                primary_cam_id = cand
                break
        if primary_cam_id is None and camera_data:
            primary_cam_id = list(camera_data.keys())[0]

        primary_imu_id = list(imu_data.keys())[0] if imu_data else None

        if primary_cam_id and primary_imu_id:
            cdata = camera_data[primary_cam_id]
            idata = imu_data[primary_imu_id]
            K = None
            if cdata.get("intrinsics"):
                intr = cdata["intrinsics"]
                K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
            results.append(check_t7_imu_cam_offset(
                camera_stream_id=primary_cam_id,
                frames_rgb=cdata["frames"],
                cam_timestamps_ns=cdata["device_times"],
                imu_stream_id=primary_imu_id,
                gyro_readings=idata["gyro"],
                imu_timestamps_ns=idata["device_times"],
                thresholds=self.thresholds.timing,
                K=K,
            ))

        # IMU Physical & Calibration Checks (T8, M1, M2, M3, M4)
        for stream_id, idata in imu_data.items():
            dev_times = idata["device_times"]
            is_sat = idata["is_saturated"]
            rate = idata["nominal_rate_hz"]
            accel = idata["accel"]
            gyro = idata["gyro"]

            if dev_times:
                all_timestamps.extend([dev_times[0], dev_times[-1]])

            # T8 IMU Stream Continuity & Saturation
            results.append(check_t8_imu_continuity(stream_id, dev_times, is_sat, rate))

            # M1-M4 Checks
            results.append(check_m1_imu_gravity(stream_id, accel, gyro, dev_times, self.thresholds.imu))
            results.append(check_m2_imu_static_bias(stream_id, accel, gyro, dev_times, self.thresholds.imu))
            results.append(check_m3_imu_bias_jump(stream_id, accel, gyro, dev_times))
            results.append(check_m4_imu_gravity_direction(stream_id, accel, gyro, dev_times))

        # 3. Overall Verdict
        t_start = min(all_timestamps) if all_timestamps else 0
        t_end = max(all_timestamps) if all_timestamps else 1_000_000_000

        verdict = self.evaluator.evaluate(
            session_id=session.session_id,
            results=results,
            t_start_ns=t_start,
            t_end_ns=t_end,
        )

        # 4. Save results JSON
        qa_out_dir = manifest_file.parent / "qa"
        qa_out_dir.mkdir(parents=True, exist_ok=True)
        results_file = qa_out_dir / "results.json"
        with open(results_file, "w") as f:
            f.write(verdict.model_dump_json(indent=2))

        # 5. Print Rich Summary Table
        self._print_verdict_summary(verdict)

        return verdict

    def _print_verdict_summary(self, verdict: SessionVerdict) -> None:
        """Display styled summary report in terminal."""
        table = Table(title=f"RigSim QA Engine Verdict — {verdict.session_id}")
        table.add_column("Check ID", style="cyan")
        table.add_column("Stream", style="magenta")
        table.add_column("Verdict", justify="center")
        table.add_column("Metric Value", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Explanation", style="dim")

        for res in verdict.results:
            if res.verdict == QAVerdict.PASS:
                v_str = "[bold green]PASS[/bold green]"
            elif res.verdict == QAVerdict.WARN:
                v_str = "[bold yellow]WARN[/bold yellow]"
            else:
                v_str = "[bold red]FAIL[/bold red]"

            table.add_row(
                res.check_id,
                res.stream_id,
                v_str,
                f"{res.value:.3f}",
                f"{res.threshold:.3f}",
                res.explanation[:60] + "..." if len(res.explanation) > 60 else res.explanation,
            )

        self.console.print(table)
        color = "green" if verdict.overall_verdict == QAVerdict.PASS else ("yellow" if verdict.overall_verdict == QAVerdict.WARN else "red")
        self.console.print(f"\n[bold {color}]Overall Session Verdict: {verdict.overall_verdict}[/bold {color}]")
        self.console.print(f"Data Yield: {verdict.yield_ratio * 100:.1f}% ({verdict.usable_duration_s:.1f}s / {verdict.total_duration_s:.1f}s usable)")
