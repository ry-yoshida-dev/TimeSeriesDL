from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..array_types import FloatArray


@dataclass(frozen=True, eq=False)
class SeriesCollection:
    """
    Validated multivariate time series that share their channels.

    Attributes
    ----------
    series : tuple[FloatArray, ...]
        Each series with shape (length, channels), as float32.

    Raises
    ------
    TypeError
        If a series is not numeric.
    ValueError
        If there is no series, a series is not 2-D or is empty, the channels differ, or a value is not finite.
    """

    series: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if not self.series:
            raise ValueError("at least one series is required.")
        converted: list[FloatArray] = []
        for position, values in enumerate(self.series):
            array: FloatArray = np.asarray(values)
            if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
                raise TypeError(f"series {position} must be numeric, got dtype {array.dtype}")
            if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
                raise ValueError(f"series {position} must have shape (length, channels), got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"series {position} contains NaN or infinite values.")
            converted.append(array.astype(np.float32))
        channels: set[int] = {array.shape[1] for array in converted}
        if len(channels) != 1:
            raise ValueError(f"every series must have the same number of channels, got {sorted(channels)}")
        object.__setattr__(self, "series", tuple(converted))

    @classmethod
    def from_input(cls, series: FloatArray | Sequence[FloatArray]) -> SeriesCollection:
        """
        Validate one series or several series.

        Parameters
        ----------
        series : FloatArray | Sequence[FloatArray]
            One array with shape (length, channels), or a sequence of such arrays.

        Returns
        -------
        SeriesCollection
            The validated series.
        """
        if isinstance(series, np.ndarray):
            return cls(series=(series,))
        return cls(series=tuple(series))

    @property
    def channels(self) -> int:
        """
        Get the number of channels.

        Returns
        -------
        int
            The number of channels.
        """
        return int(self.series[0].shape[1])

    @property
    def shortest_length(self) -> int:
        """
        Get the length of the shortest series.

        Returns
        -------
        int
            The number of time steps.
        """
        return min(int(values.shape[0]) for values in self.series)

    def concatenate(self) -> FloatArray:
        """
        Stack every time step of every series.

        Returns
        -------
        FloatArray
            The time steps with shape (total_length, channels).
        """
        return np.concatenate(self.series, axis=0)
