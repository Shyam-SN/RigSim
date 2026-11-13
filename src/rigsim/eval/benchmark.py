"""Benchmark suite and evaluation engine measuring QA precision, recall, confusion matrix, and temporal IoU."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rigsim.config.faults import (
    FaultInjectionSpec,
    FaultScenarioConfig,
)
from rigsim.config.qa import QAVerdict, SessionVerdict
from rigsim.config.rig import RigConfig
from rigsim.config.types import FaultType
from rigsim.eval.metrics import ClassificationMetrics, FaultClassEvaluation, compute_temporal_iou
from rigsim.qa.engine import QAEngine
from rigsim.sim.simulator import SimulationEngine

# Mapping from FaultType enum value to primary target QA check ID
FAULT_TO_CHECK_MAP: dict[str, str] = {
    FaultType.RANDOM_FRAME_DROP: "T1",
    FaultType.BURST_FRAME_DROP: "T1",
    FaultType.LOW_EFFECTIVE_FPS: "T2",
    FaultType.STREAM_DISCONNECT: "T2",
    FaultType.FPS_JITTER: "T3",
    FaultType.TIMESTAMP_NON_MONOTONIC: "T4",
    FaultType.TIMESTAMP_DUPLICATE: "T4",
    FaultType.CLOCK_STEP: "T4",
    FaultType.HOST_LATENCY_SPIKE: "T5",
    FaultType.CLOCK_OFFSET: "T6",
    FaultType.CLOCK_DRIFT: "T6",
    FaultType.IMU_CAM_OFFSET: "T7",
    FaultType.IMU_DROPOUT: "T8",
    FaultType.MOTION_BLUR: "I1",
    FaultType.OVEREXPOSURE: "I2",
    FaultType.UNDEREXPOSURE: "I2",
    FaultType.LENS_OCCLUSION: "I3",
    FaultType.FROZEN_FRAME: "I4",
    FaultType.CORRUPTED_FRAME: "I5",
    FaultType.HEAVY_NOISE: "I6",
    FaultType.CAMERA_BUMP: "C2",
    FaultType.INTRINSIC_SHIFT: "C1",
    FaultType.IMU_SATURATION: "M1",
    FaultType.IMU_AXIS_FLIP: "M4",
    FaultType.IMU_BIAS_JUMP: "M3",
    FaultType.IMU_HIGH_NOISE: "M2",
}

# 5 High-level category groupings for aggregate confusion matrix
CATEGORIES = ["Clean", "Timing", "Image", "Calibration", "IMU"]


@dataclass
class BenchmarkResult:
    suite_name: str
    total_sessions_evaluated: int
    overall_metrics: ClassificationMetrics
    category_metrics: dict[str, ClassificationMetrics]
    fault_metrics: dict[str, FaultClassEvaluation]
    confusion_matrix_5x5: np.ndarray  # [True Category, Pred Category]
    category_labels: list[str]
    macro_f1: float
    macro_precision: float
    macro_recall: float
    mean_temporal_iou: float
    session_summaries: list[dict[str, Any]]


class BenchmarkEngine:
    """Executes systematic precision, recall, confusion matrix, and temporal IoU evaluations."""

    def __init__(self, rig_config_path: Path | None = None):
        self.rig_path = rig_config_path or Path("configs/rigs/handheld_stereo_imu.yaml")

    def evaluate_sessions(
        self,
        session_dirs: list[Path],
        suite_name: str = "custom",
    ) -> BenchmarkResult:
        """Evaluate QA Engine on a pre-existing list of session folders with ground-truth manifests."""
        qa_engine = QAEngine()
        session_summaries: list[dict[str, Any]] = []

        # Overall binary stats (Defective vs Clean)
        overall_tp, overall_fp, overall_tn, overall_fn = 0, 0, 0, 0

        # Category stats
        cat_counts = {c: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for c in CATEGORIES}

        # 5x5 confusion matrix: rows = true, cols = predicted
        # index 0: Clean, 1: Timing, 2: Image, 3: Calibration, 4: IMU
        cat_idx_map = {c: i for i, c in enumerate(CATEGORIES)}
        conf_matrix = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)

        # Per-fault breakdown stats
        fault_stats: dict[str, dict[str, Any]] = {}
        for ftype, chk in FAULT_TO_CHECK_MAP.items():
            fault_stats[ftype] = {
                "check_id": chk,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "ious": [],
            }

        for s_dir in session_dirs:
            manifest_file = s_dir / "session.json"
            if not manifest_file.exists():
                continue

            with open(manifest_file) as f:
                manifest_data = json.load(f)

            # Discover ground truth faults from labels.json or defect_labels.json or session.json
            gt_faults: list[dict[str, Any]] = []
            labels_file = s_dir / "labels.json"
            defect_labels_file = s_dir / "defect_labels.json"
            if labels_file.exists():
                with open(labels_file) as lf:
                    gt_faults = json.load(lf)
            elif defect_labels_file.exists():
                with open(defect_labels_file) as dlf:
                    gt_faults = json.load(dlf)
            else:
                gt_faults = manifest_data.get("injected_faults", [])

            has_gt_fault = len(gt_faults) > 0

            # Determine true category
            if not has_gt_fault:
                true_cat = "Clean"
            else:
                first_fault_type = gt_faults[0].get("fault_type", "")
                true_cat = self._fault_type_to_category(first_fault_type)

            # Run QA Engine
            verdict: SessionVerdict = qa_engine.process_session(s_dir)
            detected_defective = verdict.overall_verdict in (QAVerdict.WARN, QAVerdict.FAIL)

            # Overall binary update
            if has_gt_fault and detected_defective:
                overall_tp += 1
            elif not has_gt_fault and detected_defective:
                overall_fp += 1
            elif not has_gt_fault and not detected_defective:
                overall_tn += 1
            elif has_gt_fault and not detected_defective:
                overall_fn += 1

            # Determine predicted category based on failing check IDs
            failing_checks = [r.check_id for r in verdict.results if r.verdict in (QAVerdict.WARN, QAVerdict.FAIL)]
            pred_cat = self._checks_to_category(failing_checks)

            # Update 5x5 confusion matrix
            r_idx = cat_idx_map[true_cat]
            c_idx = cat_idx_map[pred_cat]
            conf_matrix[r_idx, c_idx] += 1

            # Update category binary stats
            for c in CATEGORIES:
                is_c_true = true_cat == c
                is_c_pred = pred_cat == c
                if is_c_true and is_c_pred:
                    cat_counts[c]["tp"] += 1
                elif not is_c_true and is_c_pred:
                    cat_counts[c]["fp"] += 1
                elif not is_c_true and not is_c_pred:
                    cat_counts[c]["tn"] += 1
                elif is_c_true and not is_c_pred:
                    cat_counts[c]["fn"] += 1

            # Per-fault breakdown update
            gt_fault_types = {gf.get("fault_type", "") for gf in gt_faults}
            for ftype, fdata in fault_stats.items():
                chk_id = fdata["check_id"]
                check_fired = any(r.check_id.startswith(chk_id) and r.verdict in (QAVerdict.WARN, QAVerdict.FAIL) for r in verdict.results)
                is_fault_present = ftype in gt_fault_types

                if is_fault_present and check_fired:
                    fdata["tp"] += 1
                    # Temporal IoU calculation
                    gt_windows = []
                    for gf in gt_faults:
                        if gf.get("fault_type") == ftype:
                            # Parse ns or s
                            if "t_start_ns" in gf:
                                t_s = float(gf["t_start_ns"]) / 1e9
                                t_e = float(gf["t_end_ns"]) / 1e9
                            else:
                                t_s = float(gf.get("t_start_s", 0.0))
                                t_e = float(gf.get("t_end_s", 0.0))
                            gt_windows.append((t_s, t_e))

                    pred_windows = []
                    for r in verdict.results:
                        if r.check_id.startswith(chk_id) and r.flagged_windows:
                            for fw in r.flagged_windows:
                                pred_windows.append((float(fw.t_start_ns) / 1e9, float(fw.t_end_ns) / 1e9))
                    iou = compute_temporal_iou(gt_windows, pred_windows)
                    fdata["ious"].append(iou)
                elif not is_fault_present and check_fired:
                    fdata["fp"] += 1
                elif not is_fault_present and not check_fired:
                    fdata["tn"] += 1
                elif is_fault_present and not check_fired:
                    fdata["fn"] += 1

            session_summaries.append(
                {
                    "session_id": manifest_data.get("session_id", s_dir.name),
                    "true_category": true_cat,
                    "pred_category": pred_cat,
                    "gt_fault_count": len(gt_faults),
                    "verdict": verdict.overall_verdict.value,
                    "failing_checks": failing_checks,
                    "usable_percentage": verdict.usable_percentage,
                }
            )

        # Assemble metrics
        overall_m = ClassificationMetrics(tp=overall_tp, fp=overall_fp, tn=overall_tn, fn=overall_fn)
        cat_metrics = {
            c: ClassificationMetrics(
                tp=cat_counts[c]["tp"],
                fp=cat_counts[c]["fp"],
                tn=cat_counts[c]["tn"],
                fn=cat_counts[c]["fn"],
            )
            for c in CATEGORIES
        }

        fault_evals: dict[str, FaultClassEvaluation] = {}
        all_ious = []
        for ftype, fdata in fault_stats.items():
            m = ClassificationMetrics(tp=fdata["tp"], fp=fdata["fp"], tn=fdata["tn"], fn=fdata["fn"])
            mean_iou = float(np.mean(fdata["ious"])) if fdata["ious"] else (1.0 if m.tp == 0 else 0.0)
            if fdata["ious"]:
                all_ious.extend(fdata["ious"])
            fault_evals[ftype] = FaultClassEvaluation(
                fault_type=ftype,
                target_check_id=fdata["check_id"],
                metrics=m,
                mean_temporal_iou=mean_iou,
                severity_curve=[],
            )

        mean_global_iou = float(np.mean(all_ious)) if all_ious else 1.0

        f1_vals = [m.f1 for m in cat_metrics.values()]
        prec_vals = [m.precision for m in cat_metrics.values()]
        rec_vals = [m.recall for m in cat_metrics.values()]

        return BenchmarkResult(
            suite_name=suite_name,
            total_sessions_evaluated=len(session_summaries),
            overall_metrics=overall_m,
            category_metrics=cat_metrics,
            fault_metrics=fault_evals,
            confusion_matrix_5x5=conf_matrix,
            category_labels=CATEGORIES,
            macro_f1=float(np.mean(f1_vals)) if f1_vals else 1.0,
            macro_precision=float(np.mean(prec_vals)) if prec_vals else 1.0,
            macro_recall=float(np.mean(rec_vals)) if rec_vals else 1.0,
            mean_temporal_iou=mean_global_iou,
            session_summaries=session_summaries,
        )

    def run_automated_suite(
        self,
        output_base_dir: Path | None = None,
        suite_type: str = "quick",
    ) -> BenchmarkResult:
        """Generate and evaluate an automated test matrix of synthetic clean and fault-injected sessions."""
        import yaml

        with open(self.rig_path) as f:
            rig_cfg = RigConfig.model_validate(yaml.safe_load(f))

        base_dir = output_base_dir or Path(tempfile.mkdtemp(prefix="rigsim_bench_suite_"))
        base_dir.mkdir(parents=True, exist_ok=True)

        session_dirs: list[Path] = []

        # 1. Clean Baseline Sessions (Negative control)
        for seed in (42, 43):
            s_dir = base_dir / f"session_clean_seed_{seed}"
            sim = SimulationEngine(
                rig_config=rig_cfg,
                fault_config=None,
                duration_s=1.0,
                seed=seed,
                output_dir=s_dir,
            )
            sim.run()
            session_dirs.append(s_dir)

        # 2. Timing Fault Sessions
        timing_scenarios = [
            (
                "frame_drop",
                FaultScenarioConfig(
                    scenario_name="frame_drop_scenario",
                    description="Random frame drops",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.RANDOM_FRAME_DROP,
                            stream_id="cam0",
                            params={"drop_probability": 0.15},
                        )
                    ],
                ),
            ),
            (
                "fps_jitter",
                FaultScenarioConfig(
                    scenario_name="fps_jitter_scenario",
                    description="FPS timing jitter",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.FPS_JITTER,
                            stream_id="cam0",
                            params={"jitter_std_ms": 12.0},
                        )
                    ],
                ),
            ),
            (
                "clock_step",
                FaultScenarioConfig(
                    scenario_name="clock_step_scenario",
                    description="Timestamp non-monotonic step",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.TIMESTAMP_NON_MONOTONIC,
                            stream_id="cam0",
                            params={"step_backward_ms": 50.0, "step_at_frame": 15},
                        )
                    ],
                ),
            ),
            (
                "latency_spike",
                FaultScenarioConfig(
                    scenario_name="latency_spike_scenario",
                    description="Host transport latency spike",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.HOST_LATENCY_SPIKE,
                            stream_id="cam0",
                            t_start_s=0.3,
                            t_end_s=0.7,
                            params={"latency_spike_ms": 45.0},
                        )
                    ],
                ),
            ),
        ]
        for name, cfg in timing_scenarios:
            s_dir = base_dir / f"session_timing_{name}"
            sim = SimulationEngine(
                rig_config=rig_cfg,
                fault_config=cfg,
                duration_s=1.0,
                seed=42,
                output_dir=s_dir,
            )
            sim.run()
            session_dirs.append(s_dir)

        # 3. Image Fault Sessions
        image_scenarios = [
            (
                "motion_blur",
                FaultScenarioConfig(
                    scenario_name="motion_blur_scenario",
                    description="Motion blur fault",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.MOTION_BLUR,
                            stream_id="cam0",
                            params={"kernel_size": 19},
                        )
                    ],
                ),
            ),
            (
                "overexposure",
                FaultScenarioConfig(
                    scenario_name="overexposure_scenario",
                    description="Overexposure fault",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.OVEREXPOSURE,
                            stream_id="cam0",
                            params={"gain": 2.5, "bias": 40.0},
                        )
                    ],
                ),
            ),
            (
                "occlusion",
                FaultScenarioConfig(
                    scenario_name="occlusion_scenario",
                    description="Lens occlusion fault",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.LENS_OCCLUSION,
                            stream_id="cam0",
                            params={"coverage_pct": 0.75, "opacity": 0.9},
                        )
                    ],
                ),
            ),
            (
                "frozen",
                FaultScenarioConfig(
                    scenario_name="frozen_scenario",
                    description="Frozen frame fault",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.FROZEN_FRAME,
                            stream_id="cam0",
                            params={"freeze_at_frame": 10, "duration_frames": 15},
                        )
                    ],
                ),
            ),
            (
                "noise",
                FaultScenarioConfig(
                    scenario_name="noise_scenario",
                    description="Heavy sensor noise fault",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.HEAVY_NOISE,
                            stream_id="cam0",
                            params={"noise_std": 35.0},
                        )
                    ],
                ),
            ),
        ]
        for name, cfg in image_scenarios:
            s_dir = base_dir / f"session_image_{name}"
            sim = SimulationEngine(
                rig_config=rig_cfg,
                fault_config=cfg,
                duration_s=1.0,
                seed=42,
                output_dir=s_dir,
            )
            sim.run()
            session_dirs.append(s_dir)

        # 4. IMU Fault Sessions
        imu_scenarios = [
            (
                "saturation",
                FaultScenarioConfig(
                    scenario_name="saturation_scenario",
                    description="IMU accelerometer saturation",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.IMU_SATURATION,
                            stream_id="imu0",
                            params={"accel_max_g": 0.5},
                        )
                    ],
                ),
            ),
            (
                "axis_flip",
                FaultScenarioConfig(
                    scenario_name="axis_flip_scenario",
                    description="IMU coordinate axis flip",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.IMU_AXIS_FLIP,
                            stream_id="imu0",
                            params={"flip_z": True},
                        )
                    ],
                ),
            ),
            (
                "bias_jump",
                FaultScenarioConfig(
                    scenario_name="bias_jump_scenario",
                    description="IMU static bias jump",
                    faults=[
                        FaultInjectionSpec(
                            fault_type=FaultType.IMU_BIAS_JUMP,
                            stream_id="imu0",
                            params={"jump_at_s": 0.5, "accel_bias_jump_m_s2": (0, 0, 4.0)},
                        )
                    ],
                ),
            ),
        ]
        for name, cfg in imu_scenarios:
            s_dir = base_dir / f"session_imu_{name}"
            sim = SimulationEngine(
                rig_config=rig_cfg,
                fault_config=cfg,
                duration_s=1.0,
                seed=42,
                output_dir=s_dir,
            )
            sim.run()
            session_dirs.append(s_dir)

        return self.evaluate_sessions(session_dirs, suite_name=f"automated_{suite_type}")

    def _fault_type_to_category(self, fault_type: str) -> str:
        """Map fault string to one of the 5 top-level categories."""
        if not fault_type:
            return "Clean"
        timing_faults = {
            FaultType.RANDOM_FRAME_DROP,
            FaultType.BURST_FRAME_DROP,
            FaultType.FPS_JITTER,
            FaultType.LOW_EFFECTIVE_FPS,
            FaultType.CLOCK_DRIFT,
            FaultType.CLOCK_OFFSET,
            FaultType.IMU_CAM_OFFSET,
            FaultType.CLOCK_STEP,
            FaultType.TIMESTAMP_NON_MONOTONIC,
            FaultType.TIMESTAMP_DUPLICATE,
            FaultType.HOST_LATENCY_SPIKE,
            FaultType.IMU_DROPOUT,
            FaultType.STREAM_DISCONNECT,
        }
        image_faults = {
            FaultType.MOTION_BLUR,
            FaultType.OVEREXPOSURE,
            FaultType.UNDEREXPOSURE,
            FaultType.LENS_OCCLUSION,
            FaultType.FROZEN_FRAME,
            FaultType.CORRUPTED_FRAME,
            FaultType.HEAVY_NOISE,
        }
        calib_faults = {FaultType.CAMERA_BUMP, FaultType.INTRINSIC_SHIFT}
        imu_faults = {
            FaultType.IMU_SATURATION,
            FaultType.IMU_BIAS_JUMP,
            FaultType.IMU_HIGH_NOISE,
            FaultType.IMU_AXIS_FLIP,
        }

        if fault_type in timing_faults:
            return "Timing"
        elif fault_type in image_faults:
            return "Image"
        elif fault_type in calib_faults:
            return "Calibration"
        elif fault_type in imu_faults:
            return "IMU"
        return "Timing"

    def _checks_to_category(self, check_ids: list[str]) -> str:
        """Determine primary predicted category from triggered failing check IDs."""
        if not check_ids:
            return "Clean"
        has_timing = any(c.startswith("T") for c in check_ids)
        has_image = any(c.startswith("I") for c in check_ids)
        has_calib = any(c.startswith("C") for c in check_ids)
        has_imu = any(c.startswith("M") for c in check_ids)

        if has_timing:
            return "Timing"
        if has_image:
            return "Image"
        if has_calib:
            return "Calibration"
        if has_imu:
            return "IMU"
        return "Clean"
