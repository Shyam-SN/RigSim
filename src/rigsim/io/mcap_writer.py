"""High-performance MCAP writer for multi-camera, IMU, TF, and calibration logs."""

import json
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import google.protobuf.timestamp_pb2 as pb_ts
import numpy as np
from foxglove_schemas_protobuf import (
    CameraCalibration_pb2,
    CompressedImage_pb2,
    FrameTransform_pb2,
    Quaternion_pb2,
    Vector3_pb2,
)
from mcap.writer import Writer

from rigsim.config.camera import CameraDistortion, CameraIntrinsics
from rigsim.config.transform import Transform
from rigsim.sim.imu import IMUSample


def _ns_to_pb_timestamp(timestamp_ns: int) -> pb_ts.Timestamp:
    ts = pb_ts.Timestamp()
    ts.seconds = timestamp_ns // 1_000_000_000
    ts.nanos = timestamp_ns % 1_000_000_000
    return ts


class RigSimMCAPWriter:
    """Manages MCAP file serialization for all RigSim sensor channels and metadata."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle: BinaryIO = open(output_path, "wb")
        self.writer = Writer(self.file_handle)
        self.writer.start()

        # Cache schema and channel registrations
        self.schemas: dict[str, int] = {}
        self.channels: dict[str, int] = {}

        self._init_core_schemas()

    def _init_core_schemas(self) -> None:
        """Register Foxglove protobuf and JSON schemas."""
        # 1. CompressedImage
        self.schemas["CompressedImage"] = self.writer.register_schema(
            name=CompressedImage_pb2.CompressedImage.DESCRIPTOR.full_name,
            encoding="protobuf",
            data=CompressedImage_pb2.CompressedImage.DESCRIPTOR.file.serialized_pb,
        )

        # 2. CameraCalibration
        self.schemas["CameraCalibration"] = self.writer.register_schema(
            name=CameraCalibration_pb2.CameraCalibration.DESCRIPTOR.full_name,
            encoding="protobuf",
            data=CameraCalibration_pb2.CameraCalibration.DESCRIPTOR.file.serialized_pb,
        )

        # 3. FrameTransform
        self.schemas["FrameTransform"] = self.writer.register_schema(
            name=FrameTransform_pb2.FrameTransform.DESCRIPTOR.full_name,
            encoding="protobuf",
            data=FrameTransform_pb2.FrameTransform.DESCRIPTOR.file.serialized_pb,
        )

        # 4. IMU JSON Schema
        imu_schema = {
            "type": "object",
            "properties": {
                "linear_acceleration": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                },
                "angular_velocity": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                },
                "temperature": {"type": "number"},
                "is_saturated": {"type": "boolean"},
            },
        }
        self.schemas["ImuSample"] = self.writer.register_schema(
            name="rigsim.ImuSample",
            encoding="jsonschema",
            data=json.dumps(imu_schema).encode("utf-8"),
        )

    def register_camera_channel(self, stream_id: str) -> None:
        """Register image and calibration channels for a camera."""
        img_topic = f"/{stream_id}/image_compressed"
        self.channels[img_topic] = self.writer.register_channel(
            schema_id=self.schemas["CompressedImage"],
            topic=img_topic,
            message_encoding="protobuf",
        )

        calib_topic = f"/{stream_id}/calibration"
        self.channels[calib_topic] = self.writer.register_channel(
            schema_id=self.schemas["CameraCalibration"],
            topic=calib_topic,
            message_encoding="protobuf",
        )

    def register_imu_channel(self, stream_id: str) -> None:
        """Register IMU JSON channel."""
        imu_topic = f"/{stream_id}/data"
        self.channels[imu_topic] = self.writer.register_channel(
            schema_id=self.schemas["ImuSample"],
            topic=imu_topic,
            message_encoding="json",
        )

    def register_tf_channel(self) -> None:
        """Register TF transform channel."""
        self.channels["/tf"] = self.writer.register_channel(
            schema_id=self.schemas["FrameTransform"],
            topic="/tf",
            message_encoding="protobuf",
        )

    def write_camera_calibration(
        self,
        stream_id: str,
        frame_id: str,
        intrinsics: CameraIntrinsics,
        distortion: CameraDistortion,
        publish_time_ns: int,
        log_time_ns: int,
    ) -> None:
        """Write CameraCalibration record."""
        topic = f"/{stream_id}/calibration"
        channel_id = self.channels[topic]

        # 3x3 Intrinsic matrix K row-major
        K = [
            intrinsics.fx,
            0.0,
            intrinsics.cx,
            0.0,
            intrinsics.fy,
            intrinsics.cy,
            0.0,
            0.0,
            1.0,
        ]

        # 3x3 Identity rectification matrix R
        R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # 3x4 Projection matrix P
        P = [
            intrinsics.fx,
            0.0,
            intrinsics.cx,
            0.0,
            0.0,
            intrinsics.fy,
            intrinsics.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        msg = CameraCalibration_pb2.CameraCalibration(
            timestamp=_ns_to_pb_timestamp(publish_time_ns),
            frame_id=frame_id,
            width=intrinsics.width,
            height=intrinsics.height,
            distortion_model=distortion.model.value,
            D=distortion.coeffs,
            K=K,
            R=R,
            P=P,
        )

        self.writer.add_message(
            channel_id=channel_id,
            log_time=log_time_ns,
            publish_time=publish_time_ns,
            data=msg.SerializeToString(),
        )

    def write_image(
        self,
        stream_id: str,
        frame_id: str,
        image_rgb: np.ndarray,
        publish_time_ns: int,
        log_time_ns: int,
        encoding: str = "jpeg",
    ) -> None:
        """Compress and write image frame."""
        topic = f"/{stream_id}/image_compressed"
        channel_id = self.channels[topic]

        # Convert RGB or Grayscale to BGR for OpenCV encoder
        if image_rgb.ndim == 2:
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if encoding.lower() == "png":
            _, enc_buf = cv2.imencode(".png", bgr)
            fmt = "png"
        else:
            _, enc_buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            fmt = "jpeg"

        msg = CompressedImage_pb2.CompressedImage(
            timestamp=_ns_to_pb_timestamp(publish_time_ns),
            frame_id=frame_id,
            data=enc_buf.tobytes(),
            format=fmt,
        )

        self.writer.add_message(
            channel_id=channel_id,
            log_time=log_time_ns,
            publish_time=publish_time_ns,
            data=msg.SerializeToString(),
        )

    def write_imu(
        self,
        stream_id: str,
        sample: IMUSample,
    ) -> None:
        """Write JSON IMU sample."""
        topic = f"/{stream_id}/data"
        channel_id = self.channels[topic]

        payload = {
            "linear_acceleration": {
                "x": float(sample.linear_acceleration[0]),
                "y": float(sample.linear_acceleration[1]),
                "z": float(sample.linear_acceleration[2]),
            },
            "angular_velocity": {
                "x": float(sample.angular_velocity[0]),
                "y": float(sample.angular_velocity[1]),
                "z": float(sample.angular_velocity[2]),
            },
            "temperature": float(sample.temperature_c),
            "is_saturated": bool(sample.is_saturated),
        }

        self.writer.add_message(
            channel_id=channel_id,
            log_time=sample.host_time_ns,
            publish_time=sample.device_time_ns,
            data=json.dumps(payload).encode("utf-8"),
        )

    def write_transform(
        self,
        parent_frame: str,
        child_frame: str,
        transform: Transform,
        publish_time_ns: int,
        log_time_ns: int,
    ) -> None:
        """Write FrameTransform pose."""
        channel_id = self.channels["/tf"]

        tx, ty, tz = transform.translation
        qx, qy, qz, qw = transform.rotation_xyzw

        msg = FrameTransform_pb2.FrameTransform(
            timestamp=_ns_to_pb_timestamp(publish_time_ns),
            parent_frame_id=parent_frame,
            child_frame_id=child_frame,
            translation=Vector3_pb2.Vector3(x=tx, y=ty, z=tz),
            rotation=Quaternion_pb2.Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

        self.writer.add_message(
            channel_id=channel_id,
            log_time=log_time_ns,
            publish_time=publish_time_ns,
            data=msg.SerializeToString(),
        )

    def add_metadata(self, name: str, data: dict[str, Any]) -> None:
        """Attach session metadata record."""
        # Convert all values to string for MCAP metadata dictionary
        str_data = {k: str(v) for k, v in data.items()}
        self.writer.add_metadata(name=name, data=str_data)

    def add_attachment(
        self, name: str, media_type: str, data_bytes: bytes, log_time_ns: int = 0
    ) -> None:
        """Embed configuration or labels as attachment inside MCAP container."""
        self.writer.add_attachment(
            create_time=log_time_ns,
            log_time=log_time_ns,
            name=name,
            media_type=media_type,
            data=data_bytes,
        )

    def close(self) -> None:
        """Finalize and close MCAP file."""
        self.writer.finish()
        self.file_handle.close()
