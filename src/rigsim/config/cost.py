"""Storage and cloud cost modeling configurations."""

from pydantic import BaseModel, Field


class StoragePricingConfig(BaseModel):
    """User-editable storage pricing config (USD)."""

    hot_storage_per_gb_month: float = Field(
        0.023, gt=0.0, description="AWS S3 Standard / GCP Standard ($/GB/mo)"
    )
    cold_storage_per_gb_month: float = Field(
        0.004, gt=0.0, description="AWS S3 Glacier Instant / GCP Coldline ($/GB/mo)"
    )
    archive_storage_per_gb_month: float = Field(
        0.00099, gt=0.0, description="AWS Glacier Deep Archive ($/GB/mo)"
    )
    egress_per_gb: float = Field(0.09, ge=0.0, description="Internet data egress ($/GB)")
    cloud_gpu_encode_per_hour: float = Field(
        0.50, ge=0.0, description="Spot/On-demand GPU/CPU worker rate ($/hr)"
    )
    default_uplink_mbps: float = Field(
        100.0, gt=0.0, description="Default facility upload bandwidth (Mbps)"
    )
