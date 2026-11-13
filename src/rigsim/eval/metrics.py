"""Statistical evaluation metrics, confusion matrix, and temporal IoU calculations."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationMetrics:
    """Standard statistical classification metrics for automated QA validation."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else (1.0 if self.fp == 0 else 0.0)

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else (1.0 if self.fn == 0 else 0.0)

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return float(self.tn / denom) if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return float(2.0 * p * r / (p + r)) if (p + r) > 0.0 else 0.0

    @property
    def accuracy(self) -> float:
        return float((self.tp + self.tn) / self.total) if self.total > 0 else 1.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return float(self.fp / denom) if denom > 0 else 0.0

    @property
    def false_discovery_rate(self) -> float:
        denom = self.tp + self.fp
        return float(self.fp / denom) if denom > 0 else 0.0


def compute_temporal_iou(
    gt_windows: list[tuple[float, float]],
    pred_windows: list[tuple[float, float]],
) -> float:
    """Calculate 1D Temporal Intersection-over-Union (IoU) between ground-truth and detected defective windows.

    Windows are in seconds (t_start, t_end).
    """
    if not gt_windows and not pred_windows:
        return 1.0
    if not gt_windows or not pred_windows:
        return 0.0

    # Discretize windows into a temporal mask at 10ms resolution (100 Hz grid)
    t_min = min(min(w[0] for w in gt_windows), min(w[0] for w in pred_windows))
    t_max = max(max(w[1] for w in gt_windows), max(w[1] for w in pred_windows))

    if t_max <= t_min:
        return 1.0

    dt = 0.01  # 10 ms grid
    n_bins = int(np.ceil((t_max - t_min) / dt)) + 1
    gt_mask = np.zeros(n_bins, dtype=bool)
    pred_mask = np.zeros(n_bins, dtype=bool)

    for s, e in gt_windows:
        i_start = max(0, int(np.floor((s - t_min) / dt)))
        i_end = min(n_bins, int(np.ceil((e - t_min) / dt)))
        gt_mask[i_start:i_end] = True

    for s, e in pred_windows:
        i_start = max(0, int(np.floor((s - t_min) / dt)))
        i_end = min(n_bins, int(np.ceil((e - t_min) / dt)))
        pred_mask[i_start:i_end] = True

    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)

    return float(intersection / union) if union > 0 else 1.0


@dataclass
class FaultClassEvaluation:
    """Evaluation scorecard for an individual fault class."""

    fault_type: str
    target_check_id: str
    metrics: ClassificationMetrics
    mean_temporal_iou: float
    severity_curve: list[tuple[float, float]]  # [(severity_value, detection_rate)]
