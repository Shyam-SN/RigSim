"""Cloud storage, data transfer, and transcoding cost modeling."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from rigsim.config.cost import StoragePricingConfig
from rigsim.plan.bandwidth import BandwidthReport
from rigsim.plan.codec_bench import CodecResult


@dataclass
class CodecCostProjection:
    format_name: str
    compression_ratio: float
    total_hours: float
    storage_tb: float
    storage_gb: float
    monthly_hot_usd: float
    monthly_cold_usd: float
    monthly_archive_usd: float
    annual_hot_usd: float
    annual_cold_usd: float
    annual_archive_usd: float
    three_year_archive_usd: float
    upload_duration_hours_100mbps: float
    upload_duration_hours_1gbps: float
    upload_duration_hours_10gbps: float
    transcode_cloud_cost_usd: float
    annual_savings_vs_raw_hot_usd: float
    annual_savings_vs_raw_archive_usd: float


@dataclass
class CostModelReport:
    rig_name: str
    hours_modeled: float
    pricing: StoragePricingConfig
    projections: list[CodecCostProjection]
    executive_summary: str


class CostModel:
    """Calculates storage footprint, transfer duration, and cloud expenditure across formats."""

    def __init__(self, pricing: StoragePricingConfig | None = None):
        self.pricing = pricing or StoragePricingConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "CostModel":
        with open(path) as f:
            data = yaml.safe_load(f)
        pricing = StoragePricingConfig.model_validate(data)
        return cls(pricing=pricing)

    def evaluate(
        self,
        bandwidth: BandwidthReport,
        hours: float = 1000.0,
        codec_results: list[CodecResult] | None = None,
    ) -> CostModelReport:
        """Project multi-tier cloud storage and transfer costs for a target capture volume."""
        # Baseline raw footprint
        raw_gb_per_hour = bandwidth.total_raw_gb_per_hour
        raw_total_gb = raw_gb_per_hour * hours

        # Raw baseline reference costs
        raw_monthly_hot = raw_total_gb * self.pricing.hot_storage_per_gb_month
        raw_monthly_archive = raw_total_gb * self.pricing.archive_storage_per_gb_month
        raw_annual_hot = raw_monthly_hot * 12.0
        raw_annual_archive = raw_monthly_archive * 12.0

        projections: list[CodecCostProjection] = []

        # If empirical codec results are provided, use them; otherwise use industry default priors
        candidate_formats: list[tuple[str, float, float]] = []
        if codec_results:
            # Add Raw baseline
            candidate_formats.append(("Raw (Uncompressed RGB)", 1.0, 1.0))
            for res in codec_results:
                candidate_formats.append((f"{res.codec_name} ({res.preset_name})", res.compression_ratio, res.speedup_realtime))
        else:
            candidate_formats = [
                ("Raw (Uncompressed RGB)", 1.0, 1.0),
                ("FFV1 (Lossless Archival)", 2.8, 0.7),
                ("H.264 CRF 23 (Production)", 10.5, 3.5),
                ("HEVC / H.265 CRF 28 (Balanced)", 15.2, 2.2),
                ("SVT-AV1 CRF 28 (Next-Gen)", 19.8, 1.1),
            ]

        for fmt_name, comp_ratio, speedup in candidate_formats:
            ratio = max(1.0, comp_ratio)
            storage_gb = raw_total_gb / ratio
            storage_tb = storage_gb / 1000.0

            m_hot = storage_gb * self.pricing.hot_storage_per_gb_month
            m_cold = storage_gb * self.pricing.cold_storage_per_gb_month
            m_archive = storage_gb * self.pricing.archive_storage_per_gb_month

            ann_hot = m_hot * 12.0
            ann_cold = m_cold * 12.0
            ann_archive = m_archive * 12.0
            three_yr_archive = ann_archive * 3.0

            # Upload time: storage_gb * 8000 megabits / (bandwidth_mbps * 3600 seconds)
            up_100m = (storage_gb * 8000.0) / (100.0 * 3600.0)
            up_1g = (storage_gb * 8000.0) / (1000.0 * 3600.0)
            up_10g = (storage_gb * 8000.0) / (10000.0 * 3600.0)

            # Transcoding cost: if comp_ratio > 1.0, encoding takes hours / speedup hours of compute
            if ratio > 1.0:
                transcode_hours = hours / max(0.1, speedup)
                transcode_cost = transcode_hours * self.pricing.cloud_gpu_encode_per_hour
            else:
                transcode_cost = 0.0

            savings_hot = raw_annual_hot - ann_hot
            savings_archive = raw_annual_archive - ann_archive

            projections.append(
                CodecCostProjection(
                    format_name=fmt_name,
                    compression_ratio=ratio,
                    total_hours=hours,
                    storage_tb=storage_tb,
                    storage_gb=storage_gb,
                    monthly_hot_usd=m_hot,
                    monthly_cold_usd=m_cold,
                    monthly_archive_usd=m_archive,
                    annual_hot_usd=ann_hot,
                    annual_cold_usd=ann_cold,
                    annual_archive_usd=ann_archive,
                    three_year_archive_usd=three_yr_archive,
                    upload_duration_hours_100mbps=up_100m,
                    upload_duration_hours_1gbps=up_1g,
                    upload_duration_hours_10gbps=up_10g,
                    transcode_cloud_cost_usd=transcode_cost,
                    annual_savings_vs_raw_hot_usd=savings_hot,
                    annual_savings_vs_raw_archive_usd=savings_archive,
                )
            )

        # Generate executive summary
        best_comp = max(projections, key=lambda p: p.compression_ratio)
        exec_summary = (
            f"For {hours:.0f} hours of '{bandwidth.rig_name}' capture, raw uncompressed storage requires "
            f"{projections[0].storage_tb:.1f} TB (${projections[0].annual_hot_usd:,.2f}/yr hot, "
            f"${projections[0].annual_archive_usd:,.2f}/yr archive). "
            f"Utilizing {best_comp.format_name} reduces storage footprint to {best_comp.storage_tb:.1f} TB "
            f"({best_comp.compression_ratio:.1f}x compression), yielding annual savings of "
            f"${best_comp.annual_savings_vs_raw_hot_usd:,.2f} on Hot storage or "
            f"${best_comp.annual_savings_vs_raw_archive_usd:,.2f} on Glacier Archive. "
            f"Upload duration drops from {projections[0].upload_duration_hours_1gbps:.1f} hours to "
            f"{best_comp.upload_duration_hours_1gbps:.1f} hours over 1GbE commercial uplink."
        )

        return CostModelReport(
            rig_name=bandwidth.rig_name,
            hours_modeled=hours,
            pricing=self.pricing,
            projections=projections,
            executive_summary=exec_summary,
        )
