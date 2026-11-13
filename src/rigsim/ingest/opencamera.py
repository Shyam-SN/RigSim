"""OpenCamera Sensors (Android) dataset ingest adapter for RigSim."""

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


class OpenCameraSensorsAdapter:
    """Ingests recordings from OpenCamera Sensors (video with hardware-aligned frame timestamps + IMU)."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        clock_fit_slope: float = 1.0,
        clock_fit_offset_ns: int = 0,
        max_frames: int | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.clock_fit_slope = clock_fit_slope
        self.clock_fit_offset_ns = clock_fit_offset_ns
        self.max_frames = max_frames

    def ingest(self) -> Session:
        """Parse OpenCamera Sensors files and generate MCAP."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        mcap_path = self.output_dir / "recording.mcap"
        session_json_path = self.output_dir / "session.json"

        # 1. Locate video and CSV
        video_files = list(self.input_dir.glob("*.mp4")) + list(self.input_dir.glob("*.mov"))
        if not video_files:
            raise FileNotFoundError(f"No video found in: {self.input_dir}")
        video_path = video_files[0]

        imu_csvs = (
            list(self.input_dir.glob("*imu*.csv"))
            + list(self.input_dir.glob("*sensor*.csv"))
            + list(self.input_dir.glob("*.csv"))
        )
        if not imu_csvs:
            raise FileNotFoundError(f"No IMU CSV found in: {self.input_dir}")
        imu_path = imu_csvs[0]

        # 2. Parse IMU data
        df = pd.read_csv(imu_path)
        cols = {c.lower(): c for c in df.columns}

        # Expected columns: timestamp_ns, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
        t_col = None
        for cand in ["time_ns", "timestamp", "time"]:
            if cand in cols:
                t_col = cols[cand]
                break

        if t_col is None:
            t_col = df.columns[0]

        # Convert to nanoseconds
        raw_t = df[t_col].values
        if float(raw_t[0]) < 1e12:
            time_ns = (raw_t * 1e9).astype(np.int64)
        else:
            time_ns = raw_t.astype(np.int64)

        df["time_ns"] = time_ns

        # 3. Open Video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        t_base_ns = int(df["time_ns"].iloc[0])

        fx = fy = float(width * 0.85)
        cx = float(width / 2.0)
        cy = float(height / 2.0)
        intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
        distortion = CameraDistortion(model=DistortionModel.RADTAN, coeffs=[0.0, 0.0, 0.0, 0.0])

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

        T_id = Transform.identity()
        writer.write_transform("body", "phone_cam", T_id, t_base_ns, t_base_ns)
        writer.write_transform("body", "phone_imu", T_id, t_base_ns, t_base_ns)

        events: list[tuple[int, str, Any]] = []

        # Add IMU events
        for _, row in df.iterrows():
            ts = int(row["time_ns"])
            ax = float(row.get("accel_x", row.get("ax", 0.0)))
            ay = float(row.get("accel_y", row.get("ay", 0.0)))
            az = float(row.get("accel_z", row.get("az", 9.81)))
            wx = float(row.get("gyro_x", row.get("wx", 0.0)))
            wy = float(row.get("gyro_y", row.get("wy", 0.0)))
            wz = float(row.get("gyro_z", row.get("wz", 0.0)))
            events.append((ts, "imu", (ax, ay, az, wx, wy, wz)))

        # Add camera frames
        frame_idx = 0
        cam_dt_ns = int(1e9 / fps)
        while True:
            ret, frame = cap.read()
            if not ret or (self.max_frames and frame_idx >= self.max_frames):
                break
            frame_ts = t_base_ns + frame_idx * cam_dt_ns
            events.append((frame_ts, "cam", frame))
            frame_idx += 1

        cap.release()
        events.sort(key=lambda x: x[0])

        for ts_ns, kind, payload in events:
            aligned_ts = int(ts_ns * self.clock_fit_slope + self.clock_fit_offset_ns)
            host_ts = aligned_ts + 1_500_000

            if kind == "cam":
                img_rgb = cv2.cvtColor(payload, cv2.COLOR_BGR2RGB)
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
                    linear_acceleration=np.array([ax, ay, az]),
                    angular_velocity=np.array([wx, wy, wz]),
                    orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                    temperature_c=27.0,
                    is_saturated=False,
                )
                writer.write_imu("phone_imu", sample)

        writer.add_metadata(
            name="rigsim.session",
            data={
                "source": SessionSource.PHONE.value,
                "app": "OpenCameraSensors",
                "total_frames": str(frame_idx),
            },
        )
        writer.close()

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
                extrinsics=T_id,
            ),
            Stream(
                stream_id="phone_imu",
                kind=SensorKind.IMU,
                nominal_rate_hz=100.0,
                clock_id="phone_clock",
                frame_id="phone_imu",
                extrinsics=T_id,
            ),
        ]

        session = Session(
            session_id=self.output_dir.name,
            source=SessionSource.PHONE,
            rig_config_name="opencamera_phone_rig",
            created_at=datetime.now(UTC).isoformat(),
            operator="Nothing Phone User",
            notes=f"Converted from OpenCamera Sensors at {self.input_dir}",
            streams=streams,
            mcap_path=str(mcap_path.resolve()),
        )

        with open(session_json_path, "w") as f:
            f.write(session.model_dump_json(indent=2))

        return session
