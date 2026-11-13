"""Unit tests for QA engine, verdict evaluation, and usable interval extraction."""

from rigsim.config.qa import (
    QAResult,
    QAThresholdsConfig,
    QAVerdict,
    SessionVerdict,
    TimeWindow,
)
from rigsim.qa.verdict import VerdictEvaluator


def test_usable_ranges_no_flags() -> None:
    evaluator = VerdictEvaluator(QAThresholdsConfig())
    ranges = evaluator._compute_usable_ranges(1_000_000_000, 11_000_000_000, [])
    assert len(ranges) == 1
    assert ranges[0].t_start_ns == 1_000_000_000
    assert ranges[0].t_end_ns == 11_000_000_000


def test_usable_ranges_with_single_flag() -> None:
    evaluator = VerdictEvaluator(QAThresholdsConfig())
    bad = [(4_000_000_000, 6_000_000_000)]
    ranges = evaluator._compute_usable_ranges(1_000_000_000, 11_000_000_000, bad)
    assert len(ranges) == 2
    assert ranges[0].t_start_ns == 1_000_000_000
    assert ranges[0].t_end_ns == 4_000_000_000
    assert ranges[1].t_start_ns == 6_000_000_000
    assert ranges[1].t_end_ns == 11_000_000_000


def test_usable_ranges_overlapping_flags() -> None:
    evaluator = VerdictEvaluator(QAThresholdsConfig())
    bad = [
        (4_000_000_000, 6_000_000_000),
        (5_000_000_000, 8_000_000_000),
    ]
    ranges = evaluator._compute_usable_ranges(1_000_000_000, 11_000_000_000, bad)
    assert len(ranges) == 2
    assert ranges[0].t_start_ns == 1_000_000_000
    assert ranges[0].t_end_ns == 4_000_000_000
    assert ranges[1].t_start_ns == 8_000_000_000
    assert ranges[1].t_end_ns == 11_000_000_000


def test_verdict_all_pass() -> None:
    evaluator = VerdictEvaluator(QAThresholdsConfig())
    results = [
        QAResult(
            check_id="T1_FRAME_DROPS",
            stream_id="cam0",
            verdict=QAVerdict.PASS,
            metric_name="drop_ratio",
            value=0.0,
            threshold=0.01,
            explanation="OK",
        ),
        QAResult(
            check_id="I1_BLUR",
            stream_id="cam0",
            verdict=QAVerdict.PASS,
            metric_name="sharpness",
            value=1.0,
            threshold=0.5,
            explanation="OK",
        ),
    ]
    verdict: SessionVerdict = evaluator.evaluate("sess_test", results, 0, 10_000_000_000)
    assert verdict.overall_verdict == QAVerdict.PASS
    assert verdict.yield_ratio == 1.0


def test_verdict_with_warn() -> None:
    evaluator = VerdictEvaluator(QAThresholdsConfig())
    results = [
        QAResult(
            check_id="T2_EFFECTIVE_FPS",
            stream_id="cam0",
            verdict=QAVerdict.WARN,
            metric_name="fps",
            value=28.0,
            threshold=28.5,
            explanation="Slightly low",
        )
    ]
    verdict: SessionVerdict = evaluator.evaluate("sess_test", results, 0, 10_000_000_000)
    assert verdict.overall_verdict == QAVerdict.WARN


def test_verdict_with_critical_fail() -> None:
    evaluator = VerdictEvaluator(QAThresholdsConfig())
    results = [
        QAResult(
            check_id="T1_FRAME_DROPS",
            stream_id="cam0",
            verdict=QAVerdict.FAIL,
            metric_name="drop_ratio",
            value=0.05,
            threshold=0.01,
            flagged_windows=[TimeWindow(t_start_ns=2_000_000_000, t_end_ns=4_000_000_000)],
            explanation="High drops",
        )
    ]
    verdict: SessionVerdict = evaluator.evaluate("sess_test", results, 0, 10_000_000_000)
    assert verdict.overall_verdict == QAVerdict.FAIL
    assert verdict.usable_duration_s == 8.0
    assert verdict.yield_ratio == 0.8
