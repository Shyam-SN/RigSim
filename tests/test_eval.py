"""Unit and integration tests for Milestone M9: Evaluation Engine, Metrics, Confusion Matrix & Benchmarks."""

import tempfile
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from rigsim.cli import app
from rigsim.eval import (
    BenchmarkEngine,
    BenchmarkResult,
    ClassificationMetrics,
    QAThroughputBenchmarker,
    compute_temporal_iou,
    generate_evaluation_report,
)


def test_classification_metrics_math() -> None:
    """Verify statistical accuracy, precision, recall, F1, and FPR formulas."""
    m = ClassificationMetrics(tp=18, fp=2, tn=75, fn=5)
    assert m.total == 100
    assert abs(m.precision - 0.90) < 1e-4
    assert abs(m.recall - (18 / 23)) < 1e-4
    assert abs(m.accuracy - 0.93) < 1e-4
    assert abs(m.specificity - (75 / 77)) < 1e-4
    assert abs(m.false_positive_rate - (2 / 77)) < 1e-4
    assert abs(m.false_discovery_rate - 0.10) < 1e-4

    expected_f1 = 2.0 * 0.90 * (18 / 23) / (0.90 + (18 / 23))
    assert abs(m.f1 - expected_f1) < 1e-4

    # Test edge case: all zeros (no division by zero error)
    m_zero = ClassificationMetrics(tp=0, fp=0, tn=0, fn=0)
    assert m_zero.precision == 1.0
    assert m_zero.recall == 1.0
    assert m_zero.f1 == 1.0
    assert m_zero.accuracy == 1.0
    assert m_zero.false_positive_rate == 0.0


def test_temporal_iou_calculation() -> None:
    """Verify 1D temporal Intersection-over-Union matching."""
    # 1. Exact match
    gt = [(1.0, 3.0)]
    pred = [(1.0, 3.0)]
    assert abs(compute_temporal_iou(gt, pred) - 1.0) < 0.05

    # 2. 50% overlap: [1, 3] and [2, 4] -> intersection [2, 3] (1s), union [1, 4] (3s) -> IoU ~ 0.333
    gt = [(1.0, 3.0)]
    pred = [(2.0, 4.0)]
    iou = compute_temporal_iou(gt, pred)
    assert 0.30 < iou < 0.37

    # 3. Disjoint intervals
    gt = [(1.0, 2.0)]
    pred = [(3.0, 4.0)]
    assert compute_temporal_iou(gt, pred) == 0.0

    # 4. Both empty
    assert compute_temporal_iou([], []) == 1.0

    # 5. One empty
    assert compute_temporal_iou(gt, []) == 0.0
    assert compute_temporal_iou([], pred) == 0.0


def test_qa_throughput_benchmarker() -> None:
    """Verify execution speed benchmark on image and IMU QA checks."""
    benchmarker = QAThroughputBenchmarker(num_camera_frames=6, num_imu_samples=200)
    report = benchmarker.run_benchmark()

    assert len(report.benchmarks) >= 12
    assert report.total_duration_s > 0
    assert report.average_image_fps > 0
    assert report.average_imu_samples_per_s > 0

    for b in report.benchmarks:
        assert b.samples_processed > 0
        assert b.duration_s >= 0
        assert b.throughput_rate > 0
        assert b.realtime_speedup > 0


def test_evaluation_report_generation() -> None:
    """Verify HTML evaluation report generation with Plotly confusion matrix heatmap."""
    cm = np.array(
        [
            [10, 0, 0, 0, 0],
            [0, 8, 1, 0, 0],
            [0, 0, 9, 0, 0],
            [0, 0, 0, 7, 1],
            [0, 1, 0, 0, 8],
        ]
    )
    labels = ["Clean", "Timing", "Image", "Calibration", "IMU"]

    cat_metrics = {
        c: ClassificationMetrics(tp=8, fp=1, tn=30, fn=1) for c in labels
    }
    overall_m = ClassificationMetrics(tp=32, fp=2, tn=10, fn=2)

    mock_result = BenchmarkResult(
        suite_name="test_suite",
        total_sessions_evaluated=46,
        overall_metrics=overall_m,
        category_metrics=cat_metrics,
        fault_metrics={},
        confusion_matrix_5x5=cm,
        category_labels=labels,
        macro_f1=0.88,
        macro_precision=0.89,
        macro_recall=0.88,
        mean_temporal_iou=0.82,
        session_summaries=[
            {
                "session_id": "test_sim_01",
                "true_category": "Timing",
                "pred_category": "Timing",
                "gt_fault_count": 1,
                "verdict": "FAIL",
                "failing_checks": ["T1_FRAME_DROPS"],
                "usable_percentage": 75.0,
            }
        ],
    )

    with tempfile.TemporaryDirectory(prefix="rigsim_test_eval_report_") as tmpdir:
        out_html = Path(tmpdir) / "eval_report.html"
        generated = generate_evaluation_report(mock_result, out_html)
        assert generated.exists()
        html = generated.read_text(encoding="utf-8")
        assert "QA Engine Precision & Recall Evaluation Benchmark" in html
        assert "Confusion Matrix" in html
        assert "Macro F1" in html
        assert "test_sim_01" in html


def test_cli_bench_command() -> None:
    """Verify rigsim bench CLI execution."""
    runner = CliRunner()
    result = runner.invoke(app, ["bench", "--frames", "5", "--samples", "100"])
    assert result.exit_code == 0
    assert "RigSim Automated QA Performance & Throughput Benchmark" in result.stdout
    assert "Image Checks Average" in result.stdout


def test_benchmark_engine_evaluate_sessions_mock() -> None:
    """Verify BenchmarkEngine session ledger aggregation and confusion matrix calculation."""
    import json
    from unittest.mock import patch

    from rigsim.config.qa import QAResult, QAVerdict, SessionVerdict
    from rigsim.qa.engine import QAEngine

    with tempfile.TemporaryDirectory(prefix="rigsim_test_bench_eval_") as tmpdir:
        tmp_path = Path(tmpdir)
        s1 = tmp_path / "session_clean"
        s1.mkdir()
        (s1 / "session.json").write_text(json.dumps({"session_id": "clean_01", "injected_faults": []}))

        s2 = tmp_path / "session_timing"
        s2.mkdir()
        (s2 / "session.json").write_text(
            json.dumps(
                {
                    "session_id": "timing_01",
                    "injected_faults": [
                        {
                            "fault_type": "random_frame_drop",
                            "stream_id": "cam0",
                            "t_start_s": 0.2,
                            "t_end_s": 0.6,
                        }
                    ],
                }
            )
        )

        v_clean = SessionVerdict(
            session_id="clean_01",
            overall_verdict=QAVerdict.PASS,
            usable_duration_s=1.0,
            total_duration_s=1.0,
            yield_ratio=1.0,
            results=[],
        )
        v_timing = SessionVerdict(
            session_id="timing_01",
            overall_verdict=QAVerdict.FAIL,
            usable_duration_s=0.6,
            total_duration_s=1.0,
            yield_ratio=0.6,
            results=[
                QAResult(
                    check_id="T1_FRAME_DROPS",
                    stream_id="cam0",
                    verdict=QAVerdict.FAIL,
                    metric_name="drop_pct",
                    value=0.15,
                    threshold=0.01,
                    explanation="Frame drop detected",
                )
            ],
        )

        bench = BenchmarkEngine()
        with patch.object(QAEngine, "process_session", side_effect=[v_clean, v_timing]):
            result = bench.evaluate_sessions([s1, s2], suite_name="unit_test")

        assert result.total_sessions_evaluated == 2
        assert result.overall_metrics.tp == 1
        assert result.overall_metrics.tn == 1
        assert result.overall_metrics.fp == 0
        assert result.overall_metrics.fn == 0
        assert result.overall_metrics.accuracy == 1.0
        assert result.confusion_matrix_5x5[0, 0] == 1  # Clean -> Clean
        assert result.confusion_matrix_5x5[1, 1] == 1  # Timing -> Timing


