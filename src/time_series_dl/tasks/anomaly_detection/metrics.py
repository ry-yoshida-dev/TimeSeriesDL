from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...array_types import BoolArray, FloatArray, IndexArray


@dataclass(frozen=True)
class AnomalyDetectionMetrics:
    """
    Agreement of detected anomalous time steps with labeled ones.

    The point-adjusted variants count a whole labeled anomaly segment as detected once any of its steps is detected,
    as in most reconstruction-based anomaly detection papers; they are optimistic and reported for comparison only.

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
    adjusted_precision : float
        Precision after point adjustment.
    adjusted_recall : float
        Recall after point adjustment.
    adjusted_f1 : float
        F1 after point adjustment.
    """

    num_steps: int
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    adjusted_precision: float
    adjusted_recall: float
    adjusted_f1: float

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
        is_both_present: bool = bool(np.any(is_anomalous)) and not bool(np.all(is_anomalous))
        return cls(
            num_steps=int(is_anomalous.size),
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=cls._compute_roc_auc(is_anomalous, scores) if is_both_present else None,
            adjusted_precision=adjusted_precision,
            adjusted_recall=adjusted_recall,
            adjusted_f1=adjusted_f1,
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
    def _adjust(is_anomalous: BoolArray, is_detected: BoolArray) -> BoolArray:
        is_segment_start: BoolArray = is_anomalous & ~np.concatenate([[False], is_anomalous[:-1]])
        segment_ids: IndexArray = np.cumsum(is_segment_start) * is_anomalous
        detected_segments: IndexArray = np.unique(segment_ids[is_anomalous & is_detected])
        return is_detected | (np.isin(segment_ids, detected_segments) & is_anomalous)
