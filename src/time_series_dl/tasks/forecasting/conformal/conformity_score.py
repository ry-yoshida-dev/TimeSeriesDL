from enum import Enum

import numpy as np

from ....array_types import FloatArray
from ..prediction import ForecastPrediction


class ConformityScore(Enum):
    """
    How far a true value lies outside a forecast, which conformal prediction calibrates intervals on.

    Attributes
    ----------
    ABSOLUTE_RESIDUAL
        The absolute error of the point forecast; intervals are the point forecast plus and minus the calibrated
        margin, the same width in every window. Works with every objective.
    QUANTILE_RESIDUAL
        Conformalized quantile regression (Romano et al., 2019): how far the value lies outside the forecast central
        interval, ``max(lower - y, y - upper)``; intervals are the forecast interval widened (or narrowed) by the
        calibrated margin, so they keep the forecaster's window-by-window width. Needs the quantiles of levels
        ``(1 - coverage) / 2`` and ``(1 + coverage) / 2`` among the forecast quantiles.
    """

    ABSOLUTE_RESIDUAL = "absolute_residual"
    QUANTILE_RESIDUAL = "quantile_residual"

    def require_quantiles(self, quantile_levels: tuple[float, ...], coverage: float) -> None:
        """
        Check that forecasts with the given quantile levels can be scored for an interval of the given coverage.

        Parameters
        ----------
        quantile_levels : tuple[float, ...]
            Levels of the forecast quantiles.
        coverage : float
            Nominal coverage of the interval.

        Raises
        ------
        ValueError
            If the score needs quantile levels that are not forecast.
        """
        match self:
            case ConformityScore.ABSOLUTE_RESIDUAL:
                return
            case ConformityScore.QUANTILE_RESIDUAL:
                needed: tuple[float, float] = ((1.0 - coverage) / 2.0, (1.0 + coverage) / 2.0)
                missing: list[float] = [
                    level for level in needed if not any(np.isclose(level, known) for known in quantile_levels)
                ]
                if missing:
                    raise ValueError(
                        f"the quantile residual of coverage {coverage} needs the forecast quantiles {missing}; "
                        + f"the forecaster forecasts {quantile_levels}"
                    )

    def compute(self, prediction: ForecastPrediction, targets: FloatArray, coverage: float) -> FloatArray:
        """
        Compute the score of every forecast value.

        Parameters
        ----------
        prediction : ForecastPrediction
            The forecasts.
        targets : FloatArray
            The true values with the shape of ``prediction.point``.
        coverage : float
            Nominal coverage of the interval the score calibrates.

        Returns
        -------
        FloatArray
            The scores with the shape of ``targets``.
        """
        match self:
            case ConformityScore.ABSOLUTE_RESIDUAL:
                return np.abs(targets - prediction.point).astype(np.float64)
            case ConformityScore.QUANTILE_RESIDUAL:
                lower: FloatArray = prediction.get_quantile((1.0 - coverage) / 2.0)
                upper: FloatArray = prediction.get_quantile((1.0 + coverage) / 2.0)
                return np.maximum(lower - targets, targets - upper).astype(np.float64)

    def build_interval(
        self, prediction: ForecastPrediction, coverage: float, margin: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """
        Build the calibrated interval of every forecast value.

        Parameters
        ----------
        prediction : ForecastPrediction
            The forecasts.
        coverage : float
            Nominal coverage of the interval.
        margin : FloatArray
            Calibrated quantile of the scores, broadcastable to the shape of ``prediction.point``; infinite margins
            give unbounded (positive) or empty (negative) intervals.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            The lower and upper bounds with the shape of ``prediction.point``.
        """
        match self:
            case ConformityScore.ABSOLUTE_RESIDUAL:
                point: FloatArray = prediction.point.astype(np.float64)
                return point - margin, point + margin
            case ConformityScore.QUANTILE_RESIDUAL:
                lower: FloatArray = prediction.get_quantile((1.0 - coverage) / 2.0).astype(np.float64)
                upper: FloatArray = prediction.get_quantile((1.0 + coverage) / 2.0).astype(np.float64)
                return lower - margin, upper + margin
