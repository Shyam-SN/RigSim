"""Evaluation benchmarks, metrics, and report generators for RigSim."""

from rigsim.eval.benchmark import BenchmarkEngine, BenchmarkResult
from rigsim.eval.euroc_eval import EuRoCEvaluator, OffsetSweepResult
from rigsim.eval.metrics import ClassificationMetrics, FaultClassEvaluation, compute_temporal_iou
from rigsim.eval.report import generate_evaluation_report
from rigsim.eval.throughput import CheckThroughput, QAThroughputBenchmarker, ThroughputReport

__all__ = [
    "BenchmarkEngine",
    "BenchmarkResult",
    "ClassificationMetrics",
    "FaultClassEvaluation",
    "compute_temporal_iou",
    "generate_evaluation_report",
    "QAThroughputBenchmarker",
    "ThroughputReport",
    "CheckThroughput",
    "EuRoCEvaluator",
    "OffsetSweepResult",
]
