"""Session-level QA verdict evaluation and usable episode interval extraction."""

from rigsim.config.qa import QAResult, QAThresholdsConfig, SessionVerdict, TimeWindow
from rigsim.config.types import QAVerdict


class VerdictEvaluator:
    """Aggregates stream-level QA results into a SessionVerdict with usable time ranges."""

    def __init__(self, thresholds: QAThresholdsConfig):
        self.thresholds = thresholds
        self.critical_checks = set(thresholds.critical_checks)

    def evaluate(
        self,
        session_id: str,
        results: list[QAResult],
        t_start_ns: int,
        t_end_ns: int,
    ) -> SessionVerdict:
        total_duration_s = max(0.001, (t_end_ns - t_start_ns) / 1e9)

        has_critical_fail = False
        has_warn = False
        summary_rules: list[str] = []

        bad_intervals: list[tuple[int, int]] = []

        for res in results:
            if res.verdict == QAVerdict.FAIL:
                if res.check_id in self.critical_checks:
                    has_critical_fail = True
                    summary_rules.append(f"FAIL: [{res.stream_id}] {res.explanation}")
                    # Collect flagged windows from critical failures to exclude from usable segments
                    for win in res.flagged_windows:
                        bad_intervals.append((win.t_start_ns, win.t_end_ns))
                else:
                    has_warn = True
                    summary_rules.append(f"WARN (non-critical): [{res.stream_id}] {res.explanation}")
            elif res.verdict == QAVerdict.WARN:
                has_warn = True
                summary_rules.append(f"WARN: [{res.stream_id}] {res.explanation}")

        # Overall verdict
        if has_critical_fail:
            overall_verdict = QAVerdict.FAIL
        elif has_warn:
            overall_verdict = QAVerdict.WARN
        else:
            overall_verdict = QAVerdict.PASS
            summary_rules.append("PASS: All critical and operational data quality thresholds satisfied.")

        # Compute usable time ranges by subtracting bad intervals from [t_start_ns, t_end_ns]
        usable_ranges = self._compute_usable_ranges(t_start_ns, t_end_ns, bad_intervals)
        usable_ns = sum(r.t_end_ns - r.t_start_ns for r in usable_ranges)
        usable_duration_s = usable_ns / 1e9
        yield_ratio = float(usable_duration_s / total_duration_s)

        return SessionVerdict(
            session_id=session_id,
            overall_verdict=overall_verdict,
            usable_time_ranges=usable_ranges,
            usable_duration_s=usable_duration_s,
            total_duration_s=total_duration_s,
            yield_ratio=min(1.0, max(0.0, yield_ratio)),
            summary_rules=summary_rules,
            results=results,
        )

    def _compute_usable_ranges(
        self,
        t_start_ns: int,
        t_end_ns: int,
        bad_intervals: list[tuple[int, int]],
    ) -> list[TimeWindow]:
        """Subtract union of bad intervals from [t_start_ns, t_end_ns]."""
        if not bad_intervals:
            return [TimeWindow(t_start_ns=t_start_ns, t_end_ns=t_end_ns)]

        # Merge overlapping bad intervals
        sorted_bad = sorted(bad_intervals, key=lambda x: x[0])
        merged_bad: list[list[int]] = []
        for start, end in sorted_bad:
            if not merged_bad or start > merged_bad[-1][1]:
                merged_bad.append([start, end])
            else:
                merged_bad[-1][1] = max(merged_bad[-1][1], end)

        # Subtract merged bad intervals
        usable: list[TimeWindow] = []
        curr = t_start_ns
        for b_start, b_end in merged_bad:
            if b_start > curr:
                # Minimum viable segment length: 1.0 second (1e9 ns)
                if (b_start - curr) >= 1_000_000_000:
                    usable.append(TimeWindow(t_start_ns=curr, t_end_ns=b_start))
            curr = max(curr, b_end)

        if curr < t_end_ns and (t_end_ns - curr) >= 1_000_000_000:
            usable.append(TimeWindow(t_start_ns=curr, t_end_ns=t_end_ns))

        return usable
