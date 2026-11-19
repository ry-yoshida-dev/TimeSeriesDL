from dataclasses import dataclass

from ...array_types import FloatArray
from .prediction import ForecastPrediction


@dataclass(frozen=True, eq=False)
class ForecastBacktest:
    """
    Forecasts of every window of some series together with the true continuations, in the original scale.

    Attributes
    ----------
    prediction : ForecastPrediction
        The forecasts of each window.
    targets : FloatArray
        True future target values with shape (num_windows, horizon, target_channels).
    """

    prediction: ForecastPrediction
    targets: FloatArray
