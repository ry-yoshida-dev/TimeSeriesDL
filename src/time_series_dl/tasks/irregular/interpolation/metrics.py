from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ....array_types import BoolArray, FloatArray
from ....data.irregular import IrregularSequenceCollection


@dataclass(frozen=True, kw_only=True)
class InterpolationMetrics:
    """
    Errors of predicted values over every observed target value, in the original scale.

    Attributes
    ----------
    mean_absolute_error : float
        Mean absolute error.
    mean_squared_error : float
        Mean squared error.
    root_mean_squared_error : float
        Square root of the mean squared error.
    num_values : int
        Number of observed target values.
    """

    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    num_values: int

    @staticmethod
    def compute(predictions: Sequence[FloatArray], targets: IrregularSequenceCollection) -> InterpolationMetrics:
        """
        Measure predictions at the times of the targets.

        Parameters
        ----------
        predictions : Sequence[FloatArray]
            Predicted values of each sequence with the shape of its target values.
        targets : IrregularSequenceCollection
            The true values; NaN marks values that are not compared.

        Returns
        -------
        InterpolationMetrics
            The metrics.

        Raises
        ------
        ValueError
            If the shapes differ.
        """
        if len(predictions) != targets.num_sequences:
            raise ValueError(f"expected {targets.num_sequences} predictions, got {len(predictions)}")
        errors: list[FloatArray] = []
        for predicted, sequence in zip(predictions, targets.sequences, strict=True):
            if predicted.shape != sequence.values.shape:
                raise ValueError(f"predictions must have shape {sequence.values.shape}, got {predicted.shape}")
            is_observed: BoolArray = ~np.isnan(sequence.values)
            errors.append(predicted[is_observed] - sequence.values[is_observed])
        all_errors: FloatArray = np.concatenate(errors)
        mean_squared_error: float = float(np.mean(np.square(all_errors)))
        return InterpolationMetrics(
            mean_absolute_error=float(np.mean(np.abs(all_errors))),
            mean_squared_error=mean_squared_error,
            root_mean_squared_error=math.sqrt(mean_squared_error),
            num_values=int(all_errors.size),
        )
