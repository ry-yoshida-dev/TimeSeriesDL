from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...array_types import BoolArray, FloatArray
from .prediction import ForecastPrediction


@dataclass(frozen=True)
class ForecastMetrics:
    """
    Accuracy of forecasts in the original scale, averaged over windows, horizon steps and channels.

    Attributes
    ----------
    num_windows : int
        Number of evaluated windows.
    mae : float
        Mean absolute error.
    mse : float
        Mean squared error.
    rmse : float
        Root mean squared error.
    mape : float | None
        Mean absolute percentage error, as a fraction; None when a true value is zero.
    smape : float
        Symmetric mean absolute percentage error in [0, 2]; pairs of zeros count as no error.
    quantile_loss : float | None
        Mean pinball loss over the forecast quantile levels; None for point forecasts.
    """

    num_windows: int
    mae: float
    mse: float
    rmse: float
    mape: float | None
    smape: float
    quantile_loss: float | None

    @classmethod
    def compute(cls, prediction: ForecastPrediction, targets: FloatArray) -> ForecastMetrics:
        """
        Compute the metrics of forecasts against the true values.

        Parameters
        ----------
        prediction : ForecastPrediction
            The forecasts.
        targets : FloatArray
            The true values with the shape of ``prediction.point``.

        Returns
        -------
        ForecastMetrics
            The metrics.

        Raises
        ------
        ValueError
            If the shapes differ or there is no window.
        """
        if targets.shape != prediction.point.shape:
            raise ValueError(f"targets must have shape {prediction.point.shape}, got {targets.shape}")
        if targets.shape[0] == 0:
            raise ValueError("at least one window is required.")
        truth: FloatArray = targets.astype(np.float64)
        errors: FloatArray = prediction.point.astype(np.float64) - truth
        absolute: FloatArray = np.abs(errors)
        mse: float = float(np.mean(np.square(errors)))
        is_zero_present: bool = bool(np.any(truth == 0.0))
        denominator: FloatArray = np.abs(truth) + np.abs(prediction.point)
        is_defined: BoolArray = denominator > 0.0
        ratios: FloatArray = np.zeros_like(absolute)
        ratios[is_defined] = 2.0 * absolute[is_defined] / denominator[is_defined]
        return cls(
            num_windows=targets.shape[0],
            mae=float(np.mean(absolute)),
            mse=mse,
            rmse=float(np.sqrt(mse)),
            mape=None if is_zero_present else float(np.mean(absolute / np.abs(truth))),
            smape=float(np.mean(ratios)),
            quantile_loss=cls._compute_quantile_loss(prediction, truth),
        )

    @staticmethod
    def _compute_quantile_loss(prediction: ForecastPrediction, truth: FloatArray) -> float | None:
        if not prediction.quantile_levels:
            return None
        levels: FloatArray = np.asarray(prediction.quantile_levels, dtype=np.float64)
        errors: FloatArray = truth[..., np.newaxis] - prediction.quantiles.astype(np.float64)
        return float(np.mean(np.maximum(levels * errors, (levels - 1.0) * errors)))
