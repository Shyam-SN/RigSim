"""Master simulation engine for multi-camera and inertial data capture."""

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from scipy.spatial.transform import Rotation

from rigsim.clocks.model import ClockEngine, TransportDelayEngine
from rigsim.config.faults import FaultScenarioConfig
from rigsim.config.rig import RigConfig
from rigsim.config.session import Session, SessionSource, Stream
from rigsim.config.types import SensorKind
from rigsim.faults.engine import FaultEngine
from rigsim.io.mcap_writer import RigSimMCAPWriter
from rigsim.sim.camera import CameraSimulator
from rigsim.sim.imu import IMUSimulator
from rigsim.sim.strobe import SyncStrobeSimulator
from rigsim.sim.trajectory import TrajectoryGenerator


class SimulationEngine:
    """Orchestrates physics simulation, multi-camera rendering, IMU sampling, and MCAP logging."""

    def __init__(
        self,
        rig_config: RigConfig,
        fault_config: FaultScenarioConfig | None = None,
        duration_s: float = 10.0,
        seed: int = 42,
        output_dir: Path = Path("data/session"),
    ):
        self.rig_config = rig_config
        self.fault_config = fault_config
        self.duration_s = duration_s
        self.seed = seed
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Realistic UNIX epoch base (~Nov 2023) so uint64 MCAP timestamps accommodate negative clock offsets
        self.epoch_base_ns = 1_700_000_000_000_000_000

        # Initialize Fault Engine
        self.fault_engine = FaultEngine(
            config=fault_config,
            epoch_base_ns=self.epoch_base_ns,
            seed=seed,
        )

        # Locate MJCF asset
        asset_path = Path("assets/mjcf") / f"{rig_config.rig_name}.xml"
        if not asset_path.exists():
            raise FileNotFoundError(f"MJCF scene not found: {asset_path}")

        self.model = mujoco.MjModel.from_xml_path(str(asset_path))
        self.data = mujoco.MjData(self.model)

        # Initialize Clocks
        self.clocks: dict[str, ClockEngine] = {}
        for c in rig_config.clocks:
            self.clocks[c.clock_id] = ClockEngine(c, seed=seed, t_zero_ns=self.epoch_base_ns)
        self.transport = TransportDelayEngine(seed=seed)

        # Initialize Trajectory Generator for mocap bodies
        # Handheld rig vs tabletop wrist
        if rig_config.rig_name == "handheld_stereo_imu":
            center = (0.0, 0.0, 1.2)
            amp = (0.3, 0.3, 0.15)
        else:
            center = (0.0, -0.2, 0.95)
            amp = (0.15, 0.15, 0.08)

        self.trajectory = TrajectoryGenerator(
            duration_s=duration_s,
            dt_s=0.001,
            seed=seed,
            center_pos=center,
            pos_amplitude=amp,
        )

        # Find mocap body id
        self.mocap_body_id = -1
        for b_name in ["wrist_body", "handheld_rig"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b_name)
            if bid >= 0 and self.model.body_mocapid[bid] >= 0:
                self.mocap_body_id = self.model.body_mocapid[bid]
                break

        # Initialize Sync Strobe
        self.strobe = SyncStrobeSimulator(self.model, seed=seed)

        # Initialize IMU simulator
        if rig_config.imu is not None:
            self.imu_sim: IMUSimulator | None = IMUSimulator(rig_config.imu, seed=seed)
            accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
            gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
            quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")

            self.imu_accel_adr = self.model.sensor_adr[accel_id] if accel_id >= 0 else -1
            self.imu_gyro_adr = self.model.sensor_adr[gyro_id] if gyro_id >= 0 else -1
            self.imu_quat_adr = self.model.sensor_adr[quat_id] if quat_id >= 0 else -1
        else:
            self.imu_sim = None
            self.imu_accel_adr = -1
            self.imu_gyro_adr = -1
            self.imu_quat_adr = -1

        # Initialize Camera simulators
        self.cameras: dict[str, CameraSimulator] = {}
        for cam_spec in rig_config.cameras:
            self.cameras[cam_spec.stream_id] = CameraSimulator(cam_spec, self.model, seed=seed)

        # Single shared offscreen renderer sized to max resolution
        max_w = max(cam.resolution[0] for cam in rig_config.cameras)
        max_h = max(cam.resolution[1] for cam in rig_config.cameras)
        self.renderer = mujoco.Renderer(self.model, max_h, max_w)

    def run(self) -> Session:
        """Run simulation session and return Session manifest."""
        mcap_path = self.output_dir / "session.mcap"
        writer = RigSimMCAPWriter(mcap_path)

        # Register channels
        writer.register_tf_channel()
        for cam_id in self.cameras:
            writer.register_camera_channel(cam_id)
        if self.imu_sim:
            writer.register_imu_channel(self.rig_config.imu.stream_id)  # type: ignore[union-attr]

        # Simulation parameters
        dt_physics = 0.001  # 1 kHz physics loop
        total_steps = int(round(self.duration_s / dt_physics))

        # Camera scheduling
        # Group cameras by nominal period
        cam_schedules: dict[str, dict[str, float]] = {}
        for cam_id, cam_sim in self.cameras.items():
            cam_schedules[cam_id] = {
                "period_s": 1.0 / cam_sim.spec.nominal_rate_hz,
                "next_trigger_s": 0.0,
            }

        # IMU scheduling (default 200 Hz = 5 ms)
        imu_period_s = 1.0 / self.rig_config.imu.nominal_rate_hz if self.rig_config.imu else 1.0
        next_imu_s = 0.0

        # Calibration sent once at t=0
        for cam_id, cam_sim in self.cameras.items():
            clock = self.clocks[cam_sim.spec.clock_id]
            dev_t = clock.sim_to_device_time_ns(self.epoch_base_ns)
            host_t = self.transport.compute_host_time_ns(dev_t)
            writer.write_camera_calibration(
                stream_id=cam_id,
                frame_id=cam_sim.spec.frame_id,
                intrinsics=cam_sim.intrinsics,
                distortion=cam_sim.distortion,
                publish_time_ns=dev_t,
                log_time_ns=host_t,
            )

        # Progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"Simulating {self.rig_config.rig_name}...", total=total_steps)

            for step in range(total_steps):
                t_sim_s = step * dt_physics
                t_sim_ns = self.epoch_base_ns + int(round(t_sim_s * 1_000_000_000))

                # 1. Update kinematic mocap pose from trajectory generator
                pose = None
                if self.mocap_body_id >= 0:
                    pose = self.trajectory.get_pose(t_sim_s)
                    self.data.mocap_pos[self.mocap_body_id] = pose.position
                    self.data.mocap_quat[self.mocap_body_id] = pose.quaternion_wxyz

                # 2. Update sync strobe state
                self.strobe.update_model(t_sim_s)

                # 3. Step forward physics
                mujoco.mj_step(self.model, self.data)

                # 4. IMU sample trigger
                if self.imu_sim and t_sim_s >= (next_imu_s - 1e-6):
                    next_imu_s += imu_period_s
                    imu_stream_id = self.rig_config.imu.stream_id  # type: ignore[union-attr]

                    # Check for IMU dropout / disconnect
                    if not self.fault_engine.should_drop_imu(imu_stream_id, t_sim_ns):
                        clock = self.clocks[self.rig_config.imu.clock_id]  # type: ignore[union-attr]
                        dev_t = clock.sim_to_device_time_ns(t_sim_ns)
                        dev_t = self.fault_engine.modify_device_time(imu_stream_id, dev_t)
                        host_t = self.transport.compute_host_time_ns(dev_t)
                        host_t = self.fault_engine.modify_host_time(imu_stream_id, host_t, dev_t)

                        if self.mocap_body_id >= 0 and pose is not None:
                            gyro = pose.angular_velocity.copy()
                            q_wxyz = pose.quaternion_wxyz
                            rot = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                            lin_acc = (
                                pose.linear_acceleration
                                if pose.linear_acceleration is not None
                                else np.zeros(3)
                            )
                            accel = rot.inv().apply(lin_acc + np.array([0.0, 0.0, 9.81]))
                            quat = pose.quaternion_wxyz.copy()
                        else:
                            accel = self.data.sensordata[
                                self.imu_accel_adr : self.imu_accel_adr + 3
                            ].copy()
                            gyro = self.data.sensordata[
                                self.imu_gyro_adr : self.imu_gyro_adr + 3
                            ].copy()
                            quat = self.data.sensordata[
                                self.imu_quat_adr : self.imu_quat_adr + 4
                            ].copy()

                        imu_sample = self.imu_sim.update_and_sample(
                            true_accel=accel,
                            true_gyro=gyro,
                            true_quat_wxyz=quat,
                            device_time_ns=dev_t,
                            host_time_ns=host_t,
                            t_sim_s=t_sim_s,
                        )
                        # Apply IMU content faults
                        imu_sample = self.fault_engine.modify_imu_sample(imu_stream_id, imu_sample)
                        writer.write_imu(imu_stream_id, imu_sample)

                # 5. Camera triggers
                for cam_id, cam_sim in self.cameras.items():
                    sched = cam_schedules[cam_id]
                    if t_sim_s >= (sched["next_trigger_s"] - 1e-6):
                        sched["next_trigger_s"] += sched["period_s"]

                        # Check for frame drop or disconnect
                        if self.fault_engine.should_drop_frame(cam_id, t_sim_ns):
                            continue

                        clock = self.clocks[cam_sim.spec.clock_id]
                        dev_t = clock.sim_to_device_time_ns(t_sim_ns)
                        dev_t = self.fault_engine.modify_device_time(cam_id, dev_t)
                        host_t = self.transport.compute_host_time_ns(dev_t)
                        host_t = self.fault_engine.modify_host_time(cam_id, host_t, dev_t)

                        # Render scene from camera
                        self.renderer.update_scene(self.data, camera=cam_id)
                        raw_pixels = self.renderer.render()

                        # Crop to camera specific resolution if max resolution is larger
                        cropped = raw_pixels[: cam_sim.height, : cam_sim.width]

                        # Apply lens distortion & sensor noise
                        final_img = cam_sim.process_and_distort(cropped)

                        # Apply image content faults
                        final_img = self.fault_engine.modify_image(cam_id, final_img, dev_t)

                        # Write to MCAP
                        writer.write_image(
                            stream_id=cam_id,
                            frame_id=cam_sim.spec.frame_id,
                            image_rgb=final_img,
                            publish_time_ns=dev_t,
                            log_time_ns=host_t,
                            encoding="jpeg",
                        )

                        # Write TF pose (with possible bump modification)
                        ext = cam_sim.compute_extrinsics(self.data)
                        ext = self.fault_engine.modify_extrinsics(cam_id, ext, dev_t)
                        writer.write_transform(
                            parent_frame="world",
                            child_frame=cam_sim.spec.frame_id,
                            transform=ext,
                            publish_time_ns=dev_t,
                            log_time_ns=host_t,
                        )

                progress.update(task, advance=1)

        # Embed metadata and attachments
        meta = {
            "rig_name": self.rig_config.rig_name,
            "duration_s": self.duration_s,
            "seed": self.seed,
            "schema_version": "1.0.0",
        }
        writer.add_metadata("rigsim.session", meta)

        # Attach rig YAML
        rig_yaml_str = yaml.dump(self.rig_config.model_dump())
        writer.add_attachment(
            name="rig.yaml",
            media_type="application/x-yaml",
            data_bytes=rig_yaml_str.encode("utf-8"),
        )

        # Attach ground-truth fault labels
        labels = self.fault_engine.get_ground_truth_labels()
        labels_json_str = json.dumps([lbl.model_dump() for lbl in labels], indent=2)
        writer.add_attachment(
            name="labels.json",
            media_type="application/json",
            data_bytes=labels_json_str.encode("utf-8"),
        )

        writer.close()
        self.renderer.close()

        # Build Session manifest
        session_streams: list[Stream] = []
        for cam_id, cam_sim in self.cameras.items():
            session_streams.append(
                Stream(
                    stream_id=cam_id,
                    kind=cam_sim.spec.kind,
                    nominal_rate_hz=cam_sim.spec.nominal_rate_hz,
                    resolution=cam_sim.spec.resolution,
                    encoding="jpeg",
                    clock_id=cam_sim.spec.clock_id,
                    frame_id=cam_sim.spec.frame_id,
                    intrinsics=cam_sim.intrinsics,
                    distortion=cam_sim.distortion,
                    extrinsics=cam_sim.compute_extrinsics(self.data),
                )
            )

        if self.rig_config.imu:
            session_streams.append(
                Stream(
                    stream_id=self.rig_config.imu.stream_id,
                    kind=SensorKind.IMU,
                    nominal_rate_hz=self.rig_config.imu.nominal_rate_hz,
                    encoding="imu_v1",
                    clock_id=self.rig_config.imu.clock_id,
                    frame_id=self.rig_config.imu.frame_id,
                    extrinsics=self.rig_config.imu.extrinsics_to_rig.to_transform(),
                )
            )

        session = Session(
            session_id=f"sim_{self.rig_config.rig_name}_{self.seed}",
            source=SessionSource.SIM,
            rig_config_name=self.rig_config.rig_name,
            created_at="2026-11-13T00:00:00Z",
            operator="RigSim-Simulator",
            notes=f"Automated simulation: {self.duration_s}s, seed={self.seed}",
            streams=session_streams,
            mcap_path=str(mcap_path),
            ground_truth_labels_path=str(self.output_dir / "labels.json"),
        )

        # Save session.json and labels.json
        with open(self.output_dir / "session.json", "w") as f:
            f.write(session.model_dump_json(indent=2))

        labels_path = self.output_dir / "labels.json"
        with open(labels_path, "w") as f:
            f.write(labels_json_str)

        return session
