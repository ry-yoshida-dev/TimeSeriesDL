from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...array_types import FloatArray
from .calendar_feature import CalendarFeature


@dataclass(frozen=True)
class CalendarFeatures:
    """
    Known covariates derived from the time stamps of a series.

    Attributes
    ----------
    features : tuple[CalendarFeature, ...]
        The features, in the order of the covariate channels.

    Raises
    ------
    ValueError
        If there is no feature or a feature repeats.
    """

    features: tuple[CalendarFeature, ...] = (
        CalendarFeature.HOUR_OF_DAY,
        CalendarFeature.DAY_OF_WEEK,
        CalendarFeature.DAY_OF_MONTH,
        CalendarFeature.DAY_OF_YEAR,
    )

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("at least one calendar feature is required.")
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"features must not repeat, got {self.features}")

    def compute(self, timestamps: NDArray[np.datetime64]) -> FloatArray:
        """
        Compute the features of each time stamp.

        Parameters
        ----------
        timestamps : NDArray[np.datetime64]
            Time stamps with shape (length,); for ``Forecaster.forecast``, those of the series followed by those of the
            horizon.

        Returns
        -------
        FloatArray
            The features with shape (length, len(features)), usable as ``SeriesCovariates.known``.

        Raises
        ------
        TypeError
            If the values are not time stamps.
        ValueError
            If the time stamps are not 1-D or empty, or a time stamp is NaT.
        """
        array: NDArray[np.datetime64] = np.asarray(timestamps)
        if not np.issubdtype(array.dtype, np.datetime64):
            raise TypeError(f"timestamps must have a datetime64 dtype, got {array.dtype}")
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"timestamps must have a non-empty shape (length,), got {array.shape}")
        if np.any(np.isnat(array)):
            raise ValueError("timestamps contain NaT.")
        return np.stack([feature.encode(array) for feature in self.features], axis=1)
