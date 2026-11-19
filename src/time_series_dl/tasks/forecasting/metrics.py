from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...array_types import BoolArray, FloatArray
from .prediction import ForecastPrediction

_LEVEL_DIGITS: int = 10


@dataclass(frozen=True)
class ForecastMetrics:
    """
    Accuracy and calibration of forecasts in the original scale, averaged over windows, horizon steps and channels.

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
    crps : float | None
        Continuous ranked probability score of the sample paths, ``E|X - y| - E|X - X'| / 2`` over the empirical
        distribution of the samples; None when no samples were drawn. It equals the MAE for a point forecast.
    quantile_crps : float | None
        The CRPS approximated from the forecast quantiles, twice the mean pinball loss over their levels, which
        approaches the CRPS as the levels cover (0, 1) evenly; None for point forecasts.
    quantile_coverage : Mapping[float, float] | None
        Fraction of true values at or below each forecast quantile, by level, which a calibrated forecast matches;
        None for point forecasts.
    interval_coverage : Mapping[float, float] | None
        Fraction of true values inside each central interval, between the quantiles of levels ``tau`` and
        ``1 - tau``, by its nominal coverage ``1 - 2 tau``; None when no two levels form such a pair.
    interval_width : Mapping[float, float] | None
        Mean width of each central interval, by its nominal coverage; None when no two levels form such a pair.
    """

    num_windows: int
    mae: float
    mse: float
    rmse: float
    mape: float | None
    smape: float
    quantile_loss: float | None
    crps: float | None
    quantile_crps: float | None
    quantile_coverage: Mapping[float, float] | None
    interval_coverage: Mapping[float, float] | None
    interval_width: Mapping[float, float] | None

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
        if prediction.samples is not None and (
            prediction.samples.ndim != 4
            or prediction.samples.shape[0] != targets.shape[0]
            or prediction.samples.shape[2:] != targets.shape[1:]
        ):
            raise ValueError(
                f"samples must have shape ({targets.shape[0]}, num_samples, {targets.shape[1]}, {targets.shape[2]}), "
                + f"got {prediction.samples.shape}"
            )
        truth: FloatArray = targets.astype(np.float64)
        errors: FloatArray = prediction.point.astype(np.float64) - truth
        absolute: FloatArray = np.abs(errors)
        mse: float = float(np.mean(np.square(errors)))
        is_zero_present: bool = bool(np.any(truth == 0.0))
        denominator: FloatArray = np.abs(truth) + np.abs(prediction.point)
        is_defined: BoolArray = denominator > 0.0
        ratios: FloatArray = np.zeros_like(absolute)
        ratios[is_defined] = 2.0 * absolute[is_defined] / denominator[is_defined]
        quantile_loss: float | None = cls._compute_quantile_loss(prediction, truth)
        return cls(
            num_windows=targets.shape[0],
            mae=float(np.mean(absolute)),
            mse=mse,
            rmse=float(np.sqrt(mse)),
            mape=None if is_zero_present else float(np.mean(absolute / np.abs(truth))),
            smape=float(np.mean(ratios)),
            quantile_loss=quantile_loss,
            crps=None if prediction.samples is None else cls._compute_sample_crps(prediction.samples, truth),
            quantile_crps=None if quantile_loss is None else 2.0 * quantile_loss,
            quantile_coverage=cls._compute_quantile_coverage(prediction, truth),
            interval_coverage=cls._compute_intervals(prediction, truth, is_width=False),
            interval_width=cls._compute_intervals(prediction, truth, is_width=True),
        )

    @staticmethod
    def _compute_quantile_loss(prediction: ForecastPrediction, truth: FloatArray) -> float | None:
        if not prediction.quantile_levels:
            return None
        levels: FloatArray = np.asarray(prediction.quantile_levels, dtype=np.float64)
        errors: FloatArray = truth[..., np.newaxis] - prediction.quantiles.astype(np.float64)
        return float(np.mean(np.maximum(levels * errors, (levels - 1.0) * errors)))

    @staticmethod
    def _compute_sample_crps(samples: FloatArray, truth: FloatArray) -> float:
        values: FloatArray = np.moveaxis(samples.astype(np.float64), 1, -1)
        count: int = values.shape[-1]
        error: FloatArray = np.mean(np.abs(values - truth[..., np.newaxis]), axis=-1)
        # E|X - X'| / 2 = sum_k (2k - m - 1) x_(k) / m^2 over the sorted samples x_(1) <= ... <= x_(m).
        ranks: FloatArray = np.arange(1, count + 1, dtype=np.float64)
        spread: FloatArray = np.sum(np.sort(values, axis=-1) * (2.0 * ranks - count - 1.0), axis=-1) / count**2
        return float(np.mean(error - spread))

    @staticmethod
    def _compute_quantile_coverage(prediction: ForecastPrediction, truth: FloatArray) -> Mapping[float, float] | None:
        if not prediction.quantile_levels:
            return None
        is_below: BoolArray = truth[..., np.newaxis] <= prediction.quantiles
        fractions: FloatArray = np.mean(is_below.reshape(-1, len(prediction.quantile_levels)), axis=0)
        return {level: float(fraction) for level, fraction in zip(prediction.quantile_levels, fractions, strict=True)}

    @staticmethod
    def _compute_intervals(
        prediction: ForecastPrediction, truth: FloatArray, is_width: bool
    ) -> Mapping[float, float] | None:
        levels: tuple[float, ...] = prediction.quantile_levels
        results: dict[float, float] = {}
        for lower_index, lower_level in enumerate(levels):
            upper_matches: list[int] = [
                index for index, level in enumerate(levels) if np.isclose(level, 1.0 - lower_level)
            ]
            if lower_level >= 0.5 or not upper_matches:
                continue
            lower: FloatArray = prediction.quantiles[..., lower_index].astype(np.float64)
            upper: FloatArray = prediction.quantiles[..., upper_matches[0]].astype(np.float64)
            coverage: float = round(1.0 - 2.0 * lower_level, _LEVEL_DIGITS)
            results[coverage] = (
                float(np.mean(upper - lower)) if is_width else float(np.mean((truth >= lower) & (truth <= upper)))
            )
        return results or None
