"""EuRoC MAV ASL dataset ingest adapter and converter to RigSim Session and MCAP."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from rigsim.config.camera import CameraDistortion, CameraIntrinsics
from rigsim.config.faults import FaultScenarioConfig
from rigsim.config.session import Session, Stream
from rigsim.config.transform import Transform
from rigsim.config.types import DistortionModel, SensorKind, SessionSource
from rigsim.faults.engine import FaultEngine
from rigsim.io.mcap_writer import RigSimMCAPWriter
from rigsim.sim.imu import IMUSample

EUROC_DOWNLOAD_INSTRUCTIONS = """
[bold yellow]EuRoC MAV Dataset Directory Not Found![/bold yellow]
Please download an official EuRoC sequence (e.g. MH_01_easy or V1_01_easy) and extract it:

[bold cyan]1. Create directory:[/bold cyan]
   mkdir -p data/euroc

[bold cyan]2. Download MH_01_easy (Machine Hall 01):[/bold cyan]
   cd data/euroc
   curl -O http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
   unzip MH_01_easy.zip -d MH_01_easy

[bold cyan]3. Expected folder structure:[/bold cyan]
   data/euroc/MH_01_easy/mav0/
   ├── cam0/
   │   ├── data/
   │   ├── data.csv
   │   └── sensor.yaml
   ├── cam1/
   │   ├── data/
   │   ├── data.csv
   │   └── sensor.yaml
   └── imu0/
       ├── data.csv
       └── sensor.yaml
"""


@dataclass
class CameraSensorConfig:
    camera_model: str
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    T_BS: Transform
    rate_hz: float
    resolution: tuple[int, int]


@dataclass
class IMUSensorConfig:
    T_BS: Transform
    rate_hz: float


def parse_transform_matrix(raw: Any) -> Transform:
    """Parse T_BS transform from EuRoC sensor.yaml into a Transform object."""
    if isinstance(raw, dict) and "data" in raw:
        vals = raw["data"]
    elif isinstance(raw, list):
        vals = raw
    else:
        return Transform.identity()

    arr = np.array(vals, dtype=np.float64)
    if arr.size == 16:
        mat = arr.reshape((4, 4))
    else:
        mat = np.eye(4, dtype=np.float64)

    return Transform.from_matrix(mat)


def load_camera_sensor_yaml(yaml_path: Path) -> CameraSensorConfig:
    """Parse EuRoC cam0/cam1 sensor.yaml specification."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    res = data.get("resolution", [752, 480])
    w, h = int(res[0]), int(res[1])
    rate_hz = float(data.get("rate_hz", 20.0))

    raw_intr = data.get("intrinsics", [458.654, 457.296, 367.215, 248.375])
    intrinsics = CameraIntrinsics(
        fx=float(raw_intr[0]),
        fy=float(raw_intr[1]),
        cx=float(raw_intr[2]),
        cy=float(raw_intr[3]),
        width=w,
        height=h,
    )

    dist_model_str = str(data.get("distortion_model", "radial-tangential")).lower()
    if "rad" in dist_model_str:
        model = DistortionModel.RADTAN
    elif "equidistant" in dist_model_str or "fisheye" in dist_model_str:
        model = DistortionModel.EQUIDISTANT
    else:
        model = DistortionModel.RADTAN

    raw_coeffs = data.get("distortion_coefficients", [-0.2834, 0.0739, 0.00019, 0.000017])
    coeffs = [float(c) for c in raw_coeffs]
    distortion = CameraDistortion(model=model, coeffs=coeffs)

    T_BS = parse_transform_matrix(data.get("T_BS", {}))

    return CameraSensorConfig(
        camera_model=str(data.get("camera_model", "pinhole")),
        intrinsics=intrinsics,
        distortion=distortion,
        T_BS=T_BS,
        rate_hz=rate_hz,
        resolution=(w, h),
    )


def load_imu_sensor_yaml(yaml_path: Path) -> IMUSensorConfig:
    """Parse EuRoC imu0 sensor.yaml specification."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    rate_hz = float(data.get("rate_hz", 200.0))
    T_BS = parse_transform_matrix(data.get("T_BS", {}))

    return IMUSensorConfig(T_BS=T_BS, rate_hz=rate_hz)


class EuRoCIngestAdapter:
    """Adapts EuRoC MAV ASL datasets into RigSim Sessions, injecting faults or timestamp shifts."""

    def __init__(
        self,
        euroc_dir: Path,
        output_dir: Path,
        fault_config: FaultScenarioConfig | None = None,
        imu_time_shift_ms: float = 0.0,
        max_camera_frames: int | None = None,
        seed: int = 42,
    ):
        self.raw_dir = Path(euroc_dir)
        self.output_dir = Path(output_dir)
        self.fault_config = fault_config
        self.imu_time_shift_ms = imu_time_shift_ms
        self.max_camera_frames = max_camera_frames
        self.seed = seed

        self.mav0_dir = self._find_mav0_root(self.raw_dir)

    def _find_mav0_root(self, root: Path) -> Path:
        """Locate the mav0 directory within EuRoC folder structure."""
        if (root / "mav0" / "cam0").is_dir():
            return root / "mav0"
        if (root / "cam0").is_dir() and (root / "imu0").is_dir():
            return root
        # Check subdirectories
        for sub in root.glob("*/mav0"):
            if (sub / "cam0").is_dir():
                return sub
        for sub in root.glob("*"):
            if sub.is_dir() and (sub / "cam0").is_dir() and (sub / "imu0").is_dir():
                return sub
        raise FileNotFoundError(
            f"Could not locate EuRoC mav0 folder in '{root}'.\n{EUROC_DOWNLOAD_INSTRUCTIONS}"
        )

    def ingest(self) -> Session:
        """Convert EuRoC dataset to MCAP and generate session manifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        mcap_path = self.output_dir / "recording.mcap"
        session_json_path = self.output_dir / "session.json"
        labels_json_path = self.output_dir / "labels.json"

        # 1. Parse sensor configs
        cam0_cfg = load_camera_sensor_yaml(self.mav0_dir / "cam0" / "sensor.yaml")
        cam1_cfg = load_camera_sensor_yaml(self.mav0_dir / "cam1" / "sensor.yaml")
        imu0_cfg = load_imu_sensor_yaml(self.mav0_dir / "imu0" / "sensor.yaml")

        # 2. Read camera lists: [(ts_ns, filename, cam_id)]
        cam0_entries = self._read_camera_csv(self.mav0_dir / "cam0" / "data.csv")
        cam1_entries = self._read_camera_csv(self.mav0_dir / "cam1" / "data.csv")

        if self.max_camera_frames is not None and self.max_camera_frames > 0:
            cam0_entries = cam0_entries[: self.max_camera_frames]
            cam1_entries = cam1_entries[: self.max_camera_frames]

        # 3. Read IMU samples: [(ts_ns, wx, wy, wz, ax, ay, az)]
        imu_entries = self._read_imu_csv(self.mav0_dir / "imu0" / "data.csv")

        # Filter IMU to temporal bounding box of cameras if max_camera_frames was set
        if self.max_camera_frames is not None and cam0_entries:
            t_min = min(cam0_entries[0][0], cam1_entries[0][0]) - 50_000_000
            t_max = max(cam0_entries[-1][0], cam1_entries[-1][0]) + 50_000_000
            imu_entries = [e for e in imu_entries if t_min <= e[0] <= t_max]

        # Determine global start time for fault relative timing
        all_start_times = []
        if cam0_entries:
            all_start_times.append(cam0_entries[0][0])
        if cam1_entries:
            all_start_times.append(cam1_entries[0][0])
        if imu_entries:
            all_start_times.append(imu_entries[0][0])

        base_time_ns = min(all_start_times) if all_start_times else 0

        # 4. Initialize FaultEngine if scenario configured
        fault_engine = (
            FaultEngine(config=self.fault_config, epoch_base_ns=base_time_ns, seed=self.seed)
            if self.fault_config
            else None
        )

        # 5. Initialize MCAP Writer
        writer = RigSimMCAPWriter(mcap_path)
        writer.register_camera_channel("cam0")
        writer.register_camera_channel("cam1")
        writer.register_imu_channel("imu0")
        writer.register_tf_channel()

        # Write initial static calibrations and TFs
        writer.write_camera_calibration(
            stream_id="cam0",
            frame_id="cam0",
            intrinsics=cam0_cfg.intrinsics,
            distortion=cam0_cfg.distortion,
            publish_time_ns=base_time_ns,
            log_time_ns=base_time_ns,
        )
        writer.write_camera_calibration(
            stream_id="cam1",
            frame_id="cam1",
            intrinsics=cam1_cfg.intrinsics,
            distortion=cam1_cfg.distortion,
            publish_time_ns=base_time_ns,
            log_time_ns=base_time_ns,
        )

        writer.write_transform("body", "cam0", cam0_cfg.T_BS, base_time_ns, base_time_ns)
        writer.write_transform("body", "cam1", cam1_cfg.T_BS, base_time_ns, base_time_ns)
        writer.write_transform("body", "imu0", imu0_cfg.T_BS, base_time_ns, base_time_ns)

        # 6. Stream and interleave samples in chronological timestamp order
        # Build unified chronological event queue
        # Items: (timestamp_ns, type_tag, payload)
        # type_tag: 'cam0', 'cam1', 'imu0'
        events: list[tuple[int, str, Any]] = []
        for ts_ns, rel_fn in cam0_entries:
            events.append((ts_ns, "cam0", rel_fn))
        for ts_ns, rel_fn in cam1_entries:
            events.append((ts_ns, "cam1", rel_fn))
        for item in imu_entries:
            events.append((item[0], "imu0", item[1:]))

        events.sort(key=lambda x: x[0])

        imu_shift_ns = int(self.imu_time_shift_ms * 1_000_000)

        for ts_ns, kind, payload in events:
            # Nominal host latency transport model (1.5 ms offset with 0.1 ms jitter)
            host_ts = ts_ns + 1_500_000

            if kind in ("cam0", "cam1"):
                stream_id = kind
                rel_fn = payload
                img_path = self.mav0_dir / stream_id / "data" / rel_fn
                if not img_path.exists():
                    # Check if relative filename already includes folder
                    img_path = self.mav0_dir / stream_id / rel_fn
                if not img_path.exists():
                    continue

                if fault_engine and fault_engine.should_drop_frame(stream_id, ts_ns):
                    continue

                dev_ts = (
                    fault_engine.modify_device_time(stream_id, ts_ns) if fault_engine else ts_ns
                )
                h_ts = (
                    fault_engine.modify_host_time(stream_id, host_ts, dev_ts)
                    if fault_engine
                    else host_ts
                )

                # Load image
                raw_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if raw_img is None:
                    continue

                # Ensure 3-channel RGB representation
                if raw_img.ndim == 2:
                    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
                else:
                    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

                if fault_engine:
                    img_rgb = fault_engine.modify_image(stream_id, img_rgb, dev_ts)

                writer.write_image(
                    stream_id=stream_id,
                    frame_id=stream_id,
                    image_rgb=img_rgb,
                    publish_time_ns=dev_ts,
                    log_time_ns=h_ts,
                    encoding="jpeg",
                )

            elif kind == "imu0":
                stream_id = "imu0"
                wx, wy, wz, ax, ay, az = payload

                if fault_engine and fault_engine.should_drop_imu(stream_id, ts_ns):
                    continue

                dev_ts = (
                    fault_engine.modify_device_time(stream_id, ts_ns) if fault_engine else ts_ns
                )
                if imu_shift_ns != 0:
                    dev_ts += imu_shift_ns

                h_ts = (
                    fault_engine.modify_host_time(stream_id, host_ts, dev_ts)
                    if fault_engine
                    else host_ts
                )

                sample = IMUSample(
                    device_time_ns=dev_ts,
                    host_time_ns=h_ts,
                    linear_acceleration=np.array([ax, ay, az], dtype=np.float64),
                    angular_velocity=np.array([wx, wy, wz], dtype=np.float64),
                    orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    temperature_c=25.0,
                    is_saturated=False,
                )

                if fault_engine:
                    sample = fault_engine.modify_imu_sample(stream_id, sample)

                writer.write_imu(stream_id, sample)

        # Attach metadata
        writer.add_metadata(
            name="rigsim.session",
            data={
                "source": SessionSource.EUROC.value,
                "dataset_root": str(self.raw_dir),
                "imu_time_shift_ms": str(self.imu_time_shift_ms),
            },
        )
        writer.close()

        # 7. Build Stream definitions and Session manifest
        streams = [
            Stream(
                stream_id="cam0",
                kind=SensorKind.STEREO_LEFT,
                nominal_rate_hz=cam0_cfg.rate_hz,
                resolution=cam0_cfg.resolution,
                encoding="jpeg",
                clock_id="vi_sensor_clock",
                frame_id="cam0",
                intrinsics=cam0_cfg.intrinsics,
                distortion=cam0_cfg.distortion,
                extrinsics=cam0_cfg.T_BS,
            ),
            Stream(
                stream_id="cam1",
                kind=SensorKind.STEREO_RIGHT,
                nominal_rate_hz=cam1_cfg.rate_hz,
                resolution=cam1_cfg.resolution,
                encoding="jpeg",
                clock_id="vi_sensor_clock",
                frame_id="cam1",
                intrinsics=cam1_cfg.intrinsics,
                distortion=cam1_cfg.distortion,
                extrinsics=cam1_cfg.T_BS,
            ),
            Stream(
                stream_id="imu0",
                kind=SensorKind.IMU,
                nominal_rate_hz=imu0_cfg.rate_hz,
                clock_id="vi_sensor_clock",
                frame_id="imu0",
                extrinsics=imu0_cfg.T_BS,
            ),
        ]

        # Save fault labels if any
        ground_truth_labels_path = None
        if fault_engine and fault_engine.get_ground_truth_labels():
            labels = fault_engine.get_ground_truth_labels()
            ground_truth_labels_path = str(labels_json_path.resolve())
            with open(labels_json_path, "w") as f:
                json.dump([item.model_dump() for item in labels], f, indent=2)

        session = Session(
            session_id=self.output_dir.name,
            source=SessionSource.EUROC,
            rig_config_name="euroc_mav_vi_sensor",
            created_at=datetime.now(UTC).isoformat(),
            operator="EuRoC MAV Benchmark",
            notes=f"Converted from EuRoC dataset at {self.mav0_dir}",
            streams=streams,
            mcap_path=str(mcap_path.resolve()),
            ground_truth_labels_path=ground_truth_labels_path,
            metadata={
                "imu_time_shift_ms": self.imu_time_shift_ms,
                "has_injected_faults": fault_engine is not None,
            },
        )

        with open(session_json_path, "w") as f:
            f.write(session.model_dump_json(indent=2))

        return session

    def _read_camera_csv(self, csv_path: Path) -> list[tuple[int, str]]:
        """Parse EuRoC camera data.csv: timestamp_ns, filename."""
        entries: list[tuple[int, str]] = []
        if not csv_path.exists():
            return entries

        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    ts_ns = int(parts[0].strip())
                    fn = parts[1].strip()
                    entries.append((ts_ns, fn))

        entries.sort(key=lambda x: x[0])
        return entries

    def _read_imu_csv(
        self, csv_path: Path
    ) -> list[tuple[int, float, float, float, float, float, float]]:
        """Parse EuRoC imu0 data.csv: timestamp, w_x, w_y, w_z, a_x, a_y, a_z."""
        entries: list[tuple[int, float, float, float, float, float, float]] = []
        if not csv_path.exists():
            return entries

        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) >= 7:
                    ts_ns = int(parts[0].strip())
                    wx = float(parts[1].strip())
                    wy = float(parts[2].strip())
                    wz = float(parts[3].strip())
                    ax = float(parts[4].strip())
                    ay = float(parts[5].strip())
                    az = float(parts[6].strip())
                    entries.append((ts_ns, wx, wy, wz, ax, ay, az))

        entries.sort(key=lambda x: x[0])
        return entries
