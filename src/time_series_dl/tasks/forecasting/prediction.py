from dataclasses import dataclass

import numpy as np

from ...array_types import FloatArray


@dataclass(frozen=True, eq=False)
class ForecastPrediction:
    """
    Forecasts of a batch of windows in the original scale of the series.

    Attributes
    ----------
    point : FloatArray
        Point forecasts with shape (num_windows, horizon, target_channels).
    quantile_levels : tuple[float, ...]
        Levels of the forecast quantiles; empty for point objectives.
    quantiles : FloatArray
        Forecast quantiles with shape (num_windows, horizon, target_channels, len(quantile_levels)).
    """

    point: FloatArray
    quantile_levels: tuple[float, ...]
    quantiles: FloatArray

    def get_quantile(self, level: float) -> FloatArray:
        """
        Get the forecast quantile of one level.

        Parameters
        ----------
        level : float
            One of ``quantile_levels``.

        Returns
        -------
        FloatArray
            The quantile with shape (num_windows, horizon, target_channels).

        Raises
        ------
        KeyError
            If the level was not forecast.
        """
        matches: list[int] = [
            position for position, known in enumerate(self.quantile_levels) if np.isclose(known, level)
        ]
        if not matches:
            raise KeyError(f"quantile {level} was not forecast; available: {self.quantile_levels}")
        return self.quantiles[..., matches[0]]
