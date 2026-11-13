"""Raw bandwidth and hardware interface limit calculations for multi-sensor rigs."""

from dataclasses import dataclass

from rigsim.config.rig import RigConfig

# Standard physical bus and transport interface sustained bandwidth limits in MB/s
INTERFACE_LIMITS_MB_S = {
    "USB 2.0 (Bus limit)": 35.0,
    "USB 3.0 / 3.1 Gen 1 (Single Host Controller)": 400.0,
    "1 Gigabit Ethernet (1GbE NIC)": 110.0,
    "10 Gigabit Ethernet (10GbE NIC)": 1100.0,
    "SATA SSD (Sustained sequential write)": 450.0,
    "NVMe M.2 SSD (Sustained sequential write)": 1500.0,
    "Internet Uplink (100 Mbps Facility Fiber)": 12.5,
    "Internet Uplink (1 Gbps Commercial Fiber)": 125.0,
}


@dataclass
class CameraBandwidthSpec:
    stream_id: str
    width: int
    height: int
    fps: float
    channels: int
    bytes_per_sample: int
    raw_mb_per_s: float
    raw_gb_per_hour: float
    raw_tb_per_1000h: float


@dataclass
class IMUBandwidthSpec:
    stream_id: str
    rate_hz: float
    sample_bytes: int
    raw_mb_per_s: float
    raw_gb_per_hour: float
    raw_tb_per_1000h: float


@dataclass
class BandwidthReport:
    rig_name: str
    cameras: list[CameraBandwidthSpec]
    imu: IMUBandwidthSpec | None
    total_raw_mb_per_s: float
    total_raw_gb_per_hour: float
    total_raw_tb_per_1000h: float
    warnings: list[str]
    interface_comparisons: dict[str, dict[str, float | str | bool]]


def compute_rig_bandwidth(rig_cfg: RigConfig) -> BandwidthReport:
    """Calculate exact raw uncompressed data rates and check physical hardware bus limits."""
    cam_specs: list[CameraBandwidthSpec] = []
    total_bytes_per_sec = 0.0

    for cam in rig_cfg.cameras:
        w, h = cam.resolution
        fps = cam.nominal_rate_hz
        channels = 3  # RGB
        bytes_per_sample = 1  # 8-bit uncompressed

        # Raw uncompressed frame size
        frame_bytes = w * h * channels * bytes_per_sample
        sec_bytes = frame_bytes * fps

        mb_per_s = sec_bytes / 1e6
        gb_per_h = (sec_bytes * 3600.0) / 1e9
        tb_per_1000h = (gb_per_h * 1000.0) / 1000.0

        cam_specs.append(
            CameraBandwidthSpec(
                stream_id=cam.stream_id,
                width=w,
                height=h,
                fps=fps,
                channels=channels,
                bytes_per_sample=bytes_per_sample,
                raw_mb_per_s=mb_per_s,
                raw_gb_per_hour=gb_per_h,
                raw_tb_per_1000h=tb_per_1000h,
            )
        )
        total_bytes_per_sec += sec_bytes

    imu_spec: IMUBandwidthSpec | None = None
    if rig_cfg.imu is not None:
        rate = rig_cfg.imu.nominal_rate_hz
        # 64-bit nanosecond timestamp (8) + 3-axis accel (24) + 3-axis gyro (24) + temp (8) = 64 bytes
        sample_bytes = 64
        sec_bytes = rate * sample_bytes

        mb_per_s = sec_bytes / 1e6
        gb_per_h = (sec_bytes * 3600.0) / 1e9
        tb_per_1000h = (gb_per_h * 1000.0) / 1000.0

        imu_spec = IMUBandwidthSpec(
            stream_id="imu0",
            rate_hz=rate,
            sample_bytes=sample_bytes,
            raw_mb_per_s=mb_per_s,
            raw_gb_per_hour=gb_per_h,
            raw_tb_per_1000h=tb_per_1000h,
        )
        total_bytes_per_sec += sec_bytes

    total_mb_per_s = total_bytes_per_sec / 1e6
    total_gb_per_hour = (total_bytes_per_sec * 3600.0) / 1e9
    total_tb_per_1000h = (total_gb_per_hour * 1000.0) / 1000.0

    # Evaluate interface limits
    warnings: list[str] = []
    comparisons: dict[str, dict[str, float | str | bool]] = {}

    for iface_name, limit_mb_s in INTERFACE_LIMITS_MB_S.items():
        exceeded = total_mb_per_s > limit_mb_s
        pct = (total_mb_per_s / limit_mb_s) * 100.0
        comparisons[iface_name] = {
            "limit_mb_s": limit_mb_s,
            "utilization_pct": pct,
            "exceeded": exceeded,
        }
        if exceeded:
            warnings.append(
                f"EXCEEDED: Total raw data rate ({total_mb_per_s:.1f} MB/s) exceeds "
                f"{iface_name} sustained bandwidth ({limit_mb_s:.1f} MB/s) by {pct - 100:.1f}%. "
                f"Must compress on-device or distribute across independent host controllers."
            )

    return BandwidthReport(
        rig_name=rig_cfg.rig_name,
        cameras=cam_specs,
        imu=imu_spec,
        total_raw_mb_per_s=total_mb_per_s,
        total_raw_gb_per_hour=total_gb_per_hour,
        total_raw_tb_per_1000h=total_tb_per_1000h,
        warnings=warnings,
        interface_comparisons=comparisons,
    )
