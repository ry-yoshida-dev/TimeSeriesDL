from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.exceptions import NotFittedError

from ....array_types import BoolArray, FloatArray, IndexArray
from ....data import SeriesCovariates, WindowCovariates
from ..backtest import ForecastBacktest
from ..forecaster import Forecaster
from ..metrics import ForecastMetrics
from ..prediction import ForecastPrediction
from .settings import ConformalSettings


class ConformalForecaster:
    """
    Forecast intervals of any fitted forecaster calibrated by split conformal prediction, with adaptive conformal
    inference (Gibbs and Candès, 2021) for online updates.

    ``calibrate`` scores the forecasts of every window of held-out calibration series. Each horizon step and target
    channel gets its own interval: for nominal coverage ``1 - alpha``, the margin is the
    ``ceil((n + 1)(1 - alpha))``-th smallest of its ``n`` calibration scores, which covers a new value with probability
    at least ``1 - alpha`` when the calibration and the new windows are exchangeable. The margin is infinite when there
    are too few calibration windows for the coverage.

    Under distribution shift, ``update`` adapts the miscoverage rate ``alpha_t`` of every interval after each window
    whose true values are known: ``alpha_{t+1} = alpha_t + gamma (alpha - miss_t)``, so intervals widen after misses
    and narrow after hits, and the long-run miss rate approaches ``alpha`` for any data. The calibration scores stay
    fixed.

    Parameters
    ----------
    forecaster : Forecaster
        The forecaster whose forecasts are calibrated; fitted before ``calibrate``.
    settings : ConformalSettings
        Coverages, conformity score and adaptation rate.
    """

    def __init__(self, forecaster: Forecaster, settings: ConformalSettings) -> None:
        self.forecaster: Forecaster = forecaster
        self.settings: ConformalSettings = settings
        self._sorted_scores: FloatArray | None = None
        self._miscoverage_rates: FloatArray | None = None

    @property
    def num_calibration_windows(self) -> int:
        """
        Get the number of calibration windows.

        Returns
        -------
        int
            The number of windows scored by ``calibrate``.

        Raises
        ------
        NotFittedError
            If calibrate has not completed yet.
        """
        return int(self._require_scores().shape[0])

    @property
    def miscoverage_rates(self) -> FloatArray:
        """
        Get the current miscoverage rate of every interval, which online updates adapt.

        Returns
        -------
        FloatArray
            A copy of the rates with shape (len(coverage_levels), horizon, target_channels); ``1 - coverage`` right
            after calibration.

        Raises
        ------
        NotFittedError
            If calibrate has not completed yet.
        """
        return self._require_rates().copy()

    def calibrate(
        self,
        series: FloatArray | Sequence[FloatArray],
        covariates: SeriesCovariates | Sequence[SeriesCovariates] | None = None,
    ) -> None:
        """
        Score the forecasts of every window of held-out series, and reset the miscoverage rates to their nominal
        values.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            Calibration series with shape (length, channels), not used to fit the forecaster.
        covariates : SeriesCovariates | Sequence[SeriesCovariates] | None
            Covariates of each series, required exactly when the forecaster was fitted with covariates.

        Raises
        ------
        NotFittedError
            If the forecaster is not fitted.
        ValueError
            If a series is invalid or too short, or the conformity score needs quantiles the forecaster does not
            forecast.
        CovariateMismatchError
            If the covariates do not match those of the forecaster.
        """
        for coverage in self.settings.coverage_levels:
            self.settings.score.require_quantiles(self.forecaster.settings.objective.quantile_levels, coverage)
        result: ForecastBacktest = self.forecaster.backtest(series, covariates)
        scores: FloatArray = np.stack(
            [
                self.settings.score.compute(result.prediction, result.targets, coverage)
                for coverage in self.settings.coverage_levels
            ],
            axis=1,
        )
        self._sorted_scores = np.sort(scores, axis=0)
        nominal: FloatArray = self._nominal_rates()
        self._miscoverage_rates = np.broadcast_to(nominal, scores.shape[1:]).copy()

    def predict(self, past: FloatArray, covariates: WindowCovariates | None = None) -> ForecastPrediction:
        """
        Forecast calibrated intervals after each of a batch of past windows, with the current miscoverage rates.

        Parameters
        ----------
        past : FloatArray
            Past windows with shape (num_windows, input_length, channels) in the original scale.
        covariates : WindowCovariates | None
            Covariates of each window, as for ``Forecaster.predict``.

        Returns
        -------
        ForecastPrediction
            The forecaster's point forecast, with the interval bounds as quantiles of the levels
            ``ConformalSettings.quantile_levels``.

        Raises
        ------
        NotFittedError
            If calibrate has not completed yet.
        ValueError
            If the windows are invalid.
        CovariateMismatchError
            If the covariates do not match those of the forecaster.
        """
        rates: FloatArray = self._require_rates()
        base: ForecastPrediction = self.forecaster.predict(past, covariates)
        return self._build_prediction(base, np.broadcast_to(rates, (base.point.shape[0], *rates.shape)))

    def forecast(self, series: FloatArray, covariates: SeriesCovariates | None = None) -> ForecastPrediction:
        """
        Forecast calibrated intervals of the horizon after the end of one series.

        Parameters
        ----------
        series : FloatArray
            The series with shape (length, channels); only its last ``input_length`` steps are read.
        covariates : SeriesCovariates | None
            Covariates of the series, as for ``Forecaster.forecast``.

        Returns
        -------
        ForecastPrediction
            The forecast, with one window.

        Raises
        ------
        NotFittedError
            If calibrate has not completed yet.
        ValueError
            If the series is invalid or too short.
        CovariateMismatchError
            If the covariates do not match those of the forecaster.
        """
        rates: FloatArray = self._require_rates()
        base: ForecastPrediction = self.forecaster.forecast(series, covariates)
        return self._build_prediction(base, rates[np.newaxis])

    def update(
        self, past: FloatArray, future: FloatArray, covariates: WindowCovariates | None = None
    ) -> ForecastPrediction:
        """
        Forecast windows one after another, each with the current intervals, and adapt the miscoverage rates to
        whether each window's true values fell inside (adaptive conformal inference).

        Parameters
        ----------
        past : FloatArray
            Past windows in time order with shape (num_windows, input_length, channels).
        future : FloatArray
            True future target values of each window with shape (num_windows, horizon, target_channels), known
            once the window's horizon has passed.
        covariates : WindowCovariates | None
            Covariates of each window, as for ``Forecaster.predict``.

        Returns
        -------
        ForecastPrediction
            The intervals each window was forecast with, before its own update.

        Raises
        ------
        NotFittedError
            If calibrate has not completed yet.
        ValueError
            If the windows are invalid or the future values have the wrong shape.
        CovariateMismatchError
            If the covariates do not match those of the forecaster.
        """
        _ = self._require_rates()
        base: ForecastPrediction = self.forecaster.predict(past, covariates)
        if future.shape != base.point.shape:
            raise ValueError(f"future must have shape {base.point.shape}, got {future.shape}")
        return self._run_online(base, future.astype(np.float64))

    def evaluate(
        self,
        series: FloatArray | Sequence[FloatArray],
        covariates: SeriesCovariates | Sequence[SeriesCovariates] | None = None,
        is_online: bool = False,
    ) -> ForecastMetrics:
        """
        Measure the calibrated intervals of every window of the series against their true continuations.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One or more series with shape (length, channels).
        covariates : SeriesCovariates | Sequence[SeriesCovariates] | None
            Covariates of each series, as for ``Forecaster.evaluate``.
        is_online : bool
            Whether the windows are forecast in order with ``update``, each window's true values becoming known
            before the next window is forecast, which adapts the miscoverage rates; otherwise every window uses the
            current rates and nothing changes.

        Returns
        -------
        ForecastMetrics
            The metrics, whose ``interval_coverage`` and ``interval_width`` measure the intervals.

        Raises
        ------
        NotFittedError
            If calibrate has not completed yet.
        ValueError
            If a series is invalid or too short.
        CovariateMismatchError
            If the covariates do not match those of the forecaster.
        """
        rates: FloatArray = self._require_rates()
        result: ForecastBacktest = self.forecaster.backtest(series, covariates)
        targets: FloatArray = result.targets.astype(np.float64)
        prediction: ForecastPrediction = (
            self._run_online(result.prediction, targets)
            if is_online
            else self._build_prediction(result.prediction, np.broadcast_to(rates, (targets.shape[0], *rates.shape)))
        )
        return ForecastMetrics.compute(prediction, targets)

    def _run_online(self, base: ForecastPrediction, future: FloatArray) -> ForecastPrediction:
        rates: FloatArray = self._require_rates()
        nominal: FloatArray = self._nominal_rates()
        used: list[FloatArray] = []
        # Adaptive conformal inference is sequential: each window's rates depend on every earlier miss.
        for window in range(future.shape[0]):
            used.append(rates.copy())
            single: ForecastPrediction = self._select_window(base, window)
            lower: FloatArray
            upper: FloatArray
            lower, upper = self._compute_bounds(single, rates[np.newaxis])
            # Bounds have the levels last; the rates have them first.
            is_missed: BoolArray = (future[window] < np.moveaxis(lower[0], -1, 0)) | (
                future[window] > np.moveaxis(upper[0], -1, 0)
            )
            rates = rates + self.settings.adaptation_rate * (nominal - is_missed.astype(np.float64))
        self._miscoverage_rates = rates
        return self._build_prediction(base, np.stack(used, axis=0))

    def _build_prediction(self, base: ForecastPrediction, rates: FloatArray) -> ForecastPrediction:
        lower: FloatArray
        upper: FloatArray
        lower, upper = self._compute_bounds(base, rates)
        # Lower bounds from the widest interval inward, then upper bounds outward, matching the increasing levels.
        bounds: FloatArray = np.concatenate([lower[..., ::-1], upper], axis=-1)
        return ForecastPrediction(point=base.point, quantile_levels=self.settings.quantile_levels, quantiles=bounds)

    def _compute_bounds(self, base: ForecastPrediction, rates: FloatArray) -> tuple[FloatArray, FloatArray]:
        margins: FloatArray = self._compute_margins(rates)
        lower_bounds: list[FloatArray] = []
        upper_bounds: list[FloatArray] = []
        for index, coverage in enumerate(self.settings.coverage_levels):
            lower: FloatArray
            upper: FloatArray
            lower, upper = self.settings.score.build_interval(base, coverage, margins[:, index])
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        return np.stack(lower_bounds, axis=-1), np.stack(upper_bounds, axis=-1)

    def _compute_margins(self, rates: FloatArray) -> FloatArray:
        scores: FloatArray = self._require_scores()
        count: int = scores.shape[0]
        ranks: IndexArray = np.ceil((count + 1) * (1.0 - rates) - 1e-9).astype(np.intp)
        positions: IndexArray = np.clip(ranks - 1, 0, count - 1)
        selected: FloatArray = np.take_along_axis(scores, positions, axis=0)
        # A rank past the scores needs an unbounded interval; a rank before them (rate above 1) an empty one.
        return np.where(ranks > count, np.inf, np.where(ranks < 1, -np.inf, selected))

    def _nominal_rates(self) -> FloatArray:
        return (1.0 - np.asarray(self.settings.coverage_levels, dtype=np.float64))[:, np.newaxis, np.newaxis]

    @staticmethod
    def _select_window(prediction: ForecastPrediction, window: int) -> ForecastPrediction:
        return ForecastPrediction(
            point=prediction.point[window : window + 1],
            quantile_levels=prediction.quantile_levels,
            quantiles=prediction.quantiles[window : window + 1],
        )

    def _require_scores(self) -> FloatArray:
        if self._sorted_scores is None:
            raise NotFittedError("ConformalForecaster is not calibrated yet. Call calibrate() first.")
        return self._sorted_scores

    def _require_rates(self) -> FloatArray:
        if self._miscoverage_rates is None:
            raise NotFittedError("ConformalForecaster is not calibrated yet. Call calibrate() first.")
        return self._miscoverage_rates
