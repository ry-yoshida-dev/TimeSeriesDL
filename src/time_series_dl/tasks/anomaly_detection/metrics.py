from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...array_types import BoolArray, FloatArray, IndexArray


@dataclass(frozen=True)
class AnomalyDetectionMetrics:
    """
    Agreement of detected anomalous time steps with labeled ones, step-wise, point-adjusted and range-based.

    Point adjustment counts a whole labeled anomaly segment as detected once any of its steps is detected. It is the
    protocol of most deep anomaly detection papers (USAD, Anomaly Transformer, DCdetector, TranAD), but it is widely
    criticized: a detector that flags a few random steps reaches a high point-adjusted F1 on long segments. Report it
    only to compare with those papers, next to the step-wise and range-based metrics.

    The range-based metrics (Tatbul et al., "Precision and Recall for Time Series", NeurIPS 2018) weigh every labeled
    and every detected segment equally: recall is the mean over labeled segments of the fraction of their steps that
    are detected, and precision the mean over detected segments of the fraction of their steps that are anomalous,
    each divided by the number of segments of the other kind it overlaps (no existence reward, flat positional bias,
    reciprocal cardinality).

    Attributes
    ----------
    num_steps : int
        Number of evaluated time steps.
    precision : float
        Fraction of detected steps that are anomalous; 0 when nothing is detected.
    recall : float
        Fraction of anomalous steps that are detected; 0 when nothing is anomalous.
    f1 : float
        Harmonic mean of precision and recall.
    roc_auc : float | None
        Area under the ROC curve of the scores; None unless both normal and anomalous steps occur.
    point_adjusted_precision : float
        Precision after point adjustment; optimistic, for comparison with published results only.
    point_adjusted_recall : float
        Recall after point adjustment; optimistic, for comparison with published results only.
    point_adjusted_f1 : float
        F1 after point adjustment; optimistic, for comparison with published results only.
    range_precision : float
        Range-based precision; 0 when nothing is detected.
    range_recall : float
        Range-based recall; 0 when nothing is anomalous.
    range_f1 : float
        Harmonic mean of the range-based precision and recall.
    """

    num_steps: int
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    point_adjusted_precision: float
    point_adjusted_recall: float
    point_adjusted_f1: float
    range_precision: float
    range_recall: float
    range_f1: float

    @classmethod
    def compute(cls, is_anomalous: BoolArray, is_detected: BoolArray, scores: FloatArray) -> AnomalyDetectionMetrics:
        """
        Compute the metrics from labels, detections and scores of each time step.

        Parameters
        ----------
        is_anomalous : BoolArray
            True labels with shape (num_steps,).
        is_detected : BoolArray
            Detections with shape (num_steps,).
        scores : FloatArray
            Anomaly scores with shape (num_steps,).

        Returns
        -------
        AnomalyDetectionMetrics
            The metrics.

        Raises
        ------
        ValueError
            If the arrays are empty or their shapes differ.
        """
        if is_anomalous.ndim != 1 or is_anomalous.size == 0:
            raise ValueError(f"labels must be non-empty and 1-D, got {is_anomalous.shape}")
        if is_detected.shape != is_anomalous.shape or scores.shape != is_anomalous.shape:
            expected: str = f"detections and scores must have shape {is_anomalous.shape}"
            raise ValueError(f"{expected}, got {is_detected.shape} and {scores.shape}")
        precision: float
        recall: float
        f1: float
        precision, recall, f1 = cls._compute_scores(is_anomalous, is_detected)
        adjusted_precision: float
        adjusted_recall: float
        adjusted_f1: float
        adjusted_precision, adjusted_recall, adjusted_f1 = cls._compute_scores(
            is_anomalous, cls._adjust(is_anomalous, is_detected)
        )
        anomaly_segments: IndexArray = cls._label_segments(is_anomalous)
        detection_segments: IndexArray = cls._label_segments(is_detected)
        range_recall: float = cls._compute_range_coverage(anomaly_segments, detection_segments)
        range_precision: float = cls._compute_range_coverage(detection_segments, anomaly_segments)
        range_total: float = range_precision + range_recall
        is_both_present: bool = bool(np.any(is_anomalous)) and not bool(np.all(is_anomalous))
        return cls(
            num_steps=int(is_anomalous.size),
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=cls._compute_roc_auc(is_anomalous, scores) if is_both_present else None,
            point_adjusted_precision=adjusted_precision,
            point_adjusted_recall=adjusted_recall,
            point_adjusted_f1=adjusted_f1,
            range_precision=range_precision,
            range_recall=range_recall,
            range_f1=2.0 * range_precision * range_recall / range_total if range_total > 0.0 else 0.0,
        )

    @staticmethod
    def _compute_scores(is_anomalous: BoolArray, is_detected: BoolArray) -> tuple[float, float, float]:
        true_positives: int = int(np.count_nonzero(is_anomalous & is_detected))
        num_detected: int = int(np.count_nonzero(is_detected))
        num_anomalous: int = int(np.count_nonzero(is_anomalous))
        precision: float = true_positives / num_detected if num_detected > 0 else 0.0
        recall: float = true_positives / num_anomalous if num_anomalous > 0 else 0.0
        total: float = precision + recall
        return precision, recall, 2.0 * precision * recall / total if total > 0.0 else 0.0

    @staticmethod
    def _compute_roc_auc(is_anomalous: BoolArray, scores: FloatArray) -> float:
        # Mann-Whitney statistic with average ranks for tied scores.
        inverse: IndexArray
        counts: IndexArray
        _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
        average_ranks: FloatArray = np.cumsum(counts) - (counts - 1) / 2.0
        ranks: FloatArray = average_ranks[inverse]
        num_positive: int = int(np.count_nonzero(is_anomalous))
        num_negative: int = int(is_anomalous.size) - num_positive
        rank_sum: float = float(np.sum(ranks[is_anomalous]))
        return (rank_sum - num_positive * (num_positive + 1) / 2.0) / (num_positive * num_negative)

    @staticmethod
    def _label_segments(is_flagged: BoolArray) -> IndexArray:
        # One-based id of the run of flagged steps each step belongs to; 0 for unflagged steps.
        is_segment_start: BoolArray = is_flagged & ~np.concatenate([[False], is_flagged[:-1]])
        return np.cumsum(is_segment_start) * is_flagged

    @staticmethod
    def _compute_range_coverage(segments: IndexArray, other_segments: IndexArray) -> float:
        # Mean over the segments of the fraction of their steps covered by the other segments, divided by the number
        # of other segments each overlaps: range-based recall of labels by detections, or precision the other way.
        num_segments: int = int(segments.max())
        if num_segments == 0:
            return 0.0
        num_other: int = int(other_segments.max())
        lengths: IndexArray = np.bincount(segments, minlength=num_segments + 1)[1:]
        is_overlap: BoolArray = (segments > 0) & (other_segments > 0)
        overlapping_pairs: IndexArray = np.unique(segments[is_overlap] * (num_other + 1) + other_segments[is_overlap])
        covered: IndexArray = np.bincount(segments[is_overlap], minlength=num_segments + 1)[1:]
        num_overlapped: IndexArray = np.bincount(overlapping_pairs // (num_other + 1), minlength=num_segments + 1)[1:]
        coverage: FloatArray = covered / lengths / np.maximum(num_overlapped, 1)
        return float(coverage.mean())

    @classmethod
    def _adjust(cls, is_anomalous: BoolArray, is_detected: BoolArray) -> BoolArray:
        segment_ids: IndexArray = cls._label_segments(is_anomalous)
        detected_segments: IndexArray = np.unique(segment_ids[is_anomalous & is_detected])
        return is_detected | (np.isin(segment_ids, detected_segments) & is_anomalous)
