from dataclasses import dataclass

import numpy as np

from ..array_types import FloatArray, IndexArray


@dataclass(frozen=True)
class SlidingWindows:
    """
    Windows of fixed length taken from a series at a fixed stride, starting at its first step.

    Attributes
    ----------
    length : int
        Number of time steps per window.
    stride : int
        Number of time steps between the starts of consecutive windows.

    Raises
    ------
    ValueError
        If length or stride is not positive.
    """

    length: int
    stride: int = 1

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

    def starts(self, series_length: int) -> IndexArray:
        """
        Get the first step of every window of a series.

        Parameters
        ----------
        series_length : int
            Number of time steps of the series.

        Returns
        -------
        IndexArray
            The start of each window; empty when the series is shorter than one window.
        """
        return np.arange(0, series_length - self.length + 1, self.stride, dtype=np.intp)

    def extract(self, values: FloatArray) -> FloatArray:
        """
        Take every window of a series.

        Parameters
        ----------
        values : FloatArray
            The series with shape (length, channels).

        Returns
        -------
        FloatArray
            The windows with shape (num_windows, self.length, channels), a copy of the values.

        Raises
        ------
        ValueError
            If the series is not 2-D or shorter than one window.
        """
        if values.ndim != 2:
            raise ValueError(f"values must have shape (length, channels), got {values.shape}")
        if values.shape[0] < self.length:
            raise ValueError(f"series of length {values.shape[0]} is shorter than one window of {self.length}")
        windows: FloatArray = np.lib.stride_tricks.sliding_window_view(values, self.length, axis=0)
        return windows[:: self.stride].transpose(0, 2, 1).copy(order="C")
