"""Sensor Logger (Android) dataset ingest adapter for RigSim."""

import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from rigsim.config.camera import CameraDistortion, CameraIntrinsics
from rigsim.config.session import Session, Stream
from rigsim.config.transform import Transform
from rigsim.config.types import DistortionModel, SensorKind, SessionSource
from rigsim.io.mcap_writer import RigSimMCAPWriter
from rigsim.sim.imu import IMUSample


class SensorLoggerIngestAdapter:
    """Ingests recordings from the Android Sensor Logger app (video + accelerometer + gyroscope)."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        clock_fit_slope: float = 1.0,
        clock_fit_offset_ns: int = 0,
        max_frames: int | None = None,
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.clock_fit_slope = clock_fit_slope
        self.clock_fit_offset_ns = clock_fit_offset_ns
        self.max_frames = max_frames

    def ingest(self) -> Session:
        """Parse Sensor Logger files, merge time series, and write MCAP and session manifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        working_dir = self._prepare_directory()

        mcap_path = self.output_dir / "recording.mcap"
        session_json_path = self.output_dir / "session.json"

        # 1. Locate video and CSV files
        video_path = self._find_video_file(working_dir)
        accel_path = self._find_csv(
            working_dir, ["TotalAccelerometer.csv", "Accelerometer.csv", "accel.csv"]
        )
        gyro_path = self._find_csv(working_dir, ["Gyroscope.csv", "gyro.csv"])

        # 2. Parse IMU tables
        accel_df = self._parse_sensor_csv(accel_path)
        gyro_df = self._parse_sensor_csv(gyro_path)

        # 3. Open Video Capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video at: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # 4. Synthesize frame timestamps if video doesn't have per-frame hardware metadata
        # We anchor video start time to IMU start time or metadata start time
        t_base_ns = 0
        if not accel_df.empty:
            t_base_ns = int(accel_df["time_ns"].iloc[0])
        elif not gyro_df.empty:
            t_base_ns = int(gyro_df["time_ns"].iloc[0])
        else:
            t_base_ns = 1_700_000_000_000_000_000

        # Estimate nominal camera intrinsics for Nothing Phone 1080p
        # Approximate: f ≈ 0.8 * width
        fx = fy = float(width * 0.85)
        cx = float(width / 2.0)
        cy = float(height / 2.0)
        intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
        distortion = CameraDistortion(model=DistortionModel.RADTAN, coeffs=[0.0, 0.0, 0.0, 0.0])

        # 5. Initialize MCAP Writer
        writer = RigSimMCAPWriter(mcap_path)
        writer.register_camera_channel("phone_cam")
        writer.register_imu_channel("phone_imu")
        writer.register_tf_channel()

        writer.write_camera_calibration(
            stream_id="phone_cam",
            frame_id="phone_cam",
            intrinsics=intrinsics,
            distortion=distortion,
            publish_time_ns=t_base_ns,
            log_time_ns=t_base_ns,
        )

        T_cam = Transform.identity()
        T_imu = Transform.identity()
        writer.write_transform("body", "phone_cam", T_cam, t_base_ns, t_base_ns)
        writer.write_transform("body", "phone_imu", T_imu, t_base_ns, t_base_ns)

        # 6. Build unified chronological stream of video frames and IMU samples
        events: list[tuple[int, str, Any]] = []

        # Add IMU events
        if not accel_df.empty and not gyro_df.empty:
            # Merge / resample gyro onto accel timestamps
            merged_imu = pd.merge_asof(
                accel_df.sort_values("time_ns"),
                gyro_df.sort_values("time_ns"),
                on="time_ns",
                direction="nearest",
                suffixes=("_accel", "_gyro"),
            ).dropna()

            for _, row in merged_imu.iterrows():
                ts_ns = int(row["time_ns"])
                ax, ay, az = float(row["x_accel"]), float(row["y_accel"]), float(row["z_accel"])
                wx, wy, wz = float(row["x_gyro"]), float(row["y_gyro"]), float(row["z_gyro"])
                events.append((ts_ns, "imu", (ax, ay, az, wx, wy, wz)))

        # Add camera events
        frame_idx = 0
        cam_dt_ns = int(1_000_000_000 / fps)
        while True:
            ret, frame = cap.read()
            if not ret or (self.max_frames and frame_idx >= self.max_frames):
                break

            frame_ts_ns = t_base_ns + frame_idx * cam_dt_ns
            events.append((frame_ts_ns, "cam", frame))
            frame_idx += 1

        cap.release()

        # Sort all events chronologically
        events.sort(key=lambda x: x[0])

        for ts_ns, kind, payload in events:
            # Apply clock model if cross-device mapping configured: t_aligned = slope * t + offset
            aligned_ts = int(ts_ns * self.clock_fit_slope + self.clock_fit_offset_ns)
            host_ts = aligned_ts + 2_000_000  # 2 ms nominal host latency

            if kind == "cam":
                img_bgr = payload
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                writer.write_image(
                    stream_id="phone_cam",
                    frame_id="phone_cam",
                    image_rgb=img_rgb,
                    publish_time_ns=aligned_ts,
                    log_time_ns=host_ts,
                    encoding="jpeg",
                )
            elif kind == "imu":
                ax, ay, az, wx, wy, wz = payload
                sample = IMUSample(
                    device_time_ns=aligned_ts,
                    host_time_ns=host_ts,
                    linear_acceleration=np.array([ax, ay, az], dtype=np.float64),
                    angular_velocity=np.array([wx, wy, wz], dtype=np.float64),
                    orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    temperature_c=28.0,
                    is_saturated=False,
                )
                writer.write_imu("phone_imu", sample)

        writer.add_metadata(
            name="rigsim.session",
            data={
                "source": SessionSource.PHONE.value,
                "app": "SensorLogger",
                "total_frames": str(frame_idx),
            },
        )
        writer.close()

        # 7. Write session manifest
        streams = [
            Stream(
                stream_id="phone_cam",
                kind=SensorKind.RGB,
                nominal_rate_hz=float(fps),
                resolution=(width, height),
                encoding="jpeg",
                clock_id="phone_clock",
                frame_id="phone_cam",
                intrinsics=intrinsics,
                distortion=distortion,
                extrinsics=T_cam,
            ),
            Stream(
                stream_id="phone_imu",
                kind=SensorKind.IMU,
                nominal_rate_hz=100.0,  # Sensor Logger standard rate
                clock_id="phone_clock",
                frame_id="phone_imu",
                extrinsics=T_imu,
            ),
        ]

        session = Session(
            session_id=self.output_dir.name,
            source=SessionSource.PHONE,
            rig_config_name="nothing_phone_rig",
            created_at=datetime.now(UTC).isoformat(),
            operator="Nothing Phone User",
            notes=f"Converted from Sensor Logger recording at {self.input_path}",
            streams=streams,
            mcap_path=str(mcap_path.resolve()),
            metadata={"fps": fps, "width": width, "height": height},
        )

        with open(session_json_path, "w") as f:
            f.write(session.model_dump_json(indent=2))

        return session

    def _prepare_directory(self) -> Path:
        """Handle zipped exports or raw directories."""
        if self.input_path.is_file() and self.input_path.suffix.lower() == ".zip":
            extract_dir = self.output_dir / "unzipped_raw"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.input_path, "r") as zf:
                zf.extractall(extract_dir)
            return extract_dir
        return self.input_path

    def _find_video_file(self, root: Path) -> Path:
        """Find video file (.mp4 or .mov)."""
        for ext in ("*.mp4", "*.MOV", "*.mov", "*.m4v"):
            found = list(root.glob(ext)) + list(root.glob(f"**/{ext}"))
            if found:
                return found[0]
        raise FileNotFoundError(f"No video file found in Sensor Logger directory: {root}")

    def _find_csv(self, root: Path, candidates: list[str]) -> Path:
        """Find sensor CSV from filename candidates."""
        for c in candidates:
            p = root / c
            if p.exists():
                return p
            found = list(root.glob(f"**/{c}"))
            if found:
                return found[0]
        raise FileNotFoundError(f"Could not locate sensor CSV matching {candidates} in: {root}")

    def _parse_sensor_csv(self, csv_path: Path) -> pd.DataFrame:
        """Parse Sensor Logger CSV with robust timestamp handling."""
        df = pd.read_csv(csv_path)

        # Standardize columns
        cols = {c.lower(): c for c in df.columns}
        t_col = None
        for cand in ["time", "timestamp", "seconds_elapsed", "time_ns"]:
            if cand in cols:
                t_col = cols[cand]
                break

        if t_col is None:
            raise ValueError(f"No timestamp column identified in {csv_path}")

        # Convert timestamps to nanoseconds
        raw_vals = df[t_col].values
        if np.issubdtype(raw_vals.dtype, np.number):
            first_val = float(raw_vals[0])
            if first_val < 1e6:
                # Value is in seconds elapsed
                time_ns = (raw_vals * 1e9).astype(np.int64)
            elif first_val < 1e12:
                # Value is Unix seconds
                time_ns = (raw_vals * 1e9).astype(np.int64)
            else:
                # Value is in nanoseconds
                time_ns = raw_vals.astype(np.int64)
        else:
            # Datetime strings
            dt_series = pd.to_datetime(df[t_col])
            time_ns = dt_series.astype(np.int64).values

        df["time_ns"] = time_ns
        return df[["time_ns", cols.get("x", "x"), cols.get("y", "y"), cols.get("z", "z")]].rename(
            columns={cols.get("x", "x"): "x", cols.get("y", "y"): "y", cols.get("z", "z"): "z"}
        )
