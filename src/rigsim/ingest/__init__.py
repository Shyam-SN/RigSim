"""Ingest adapters for external datasets and hardware recording tools."""

from rigsim.ingest.euroc import EUROC_DOWNLOAD_INSTRUCTIONS, EuRoCIngestAdapter
from rigsim.ingest.opencamera import OpenCameraSensorsAdapter
from rigsim.ingest.sensor_logger import SensorLoggerIngestAdapter

__all__ = [
    "EuRoCIngestAdapter",
    "EUROC_DOWNLOAD_INSTRUCTIONS",
    "SensorLoggerIngestAdapter",
    "OpenCameraSensorsAdapter",
]
