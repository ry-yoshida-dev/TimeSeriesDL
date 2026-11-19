from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...array_types import FloatArray, IndexArray


@dataclass(frozen=True, eq=False)
class QueryTimes:
    """
    Times at which the values of each irregularly sampled sequence are predicted.

    Attributes
    ----------
    times : tuple[FloatArray, ...]
        Non-decreasing finite times of each sequence with shape (num_queries,), as float64.

    Raises
    ------
    TypeError
        If times are not numeric.
    ValueError
        If there are no sequences, or the times of a sequence are empty, not 1-D, not finite or decreasing.
    """

    times: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("times must not be empty.")
        validated: list[FloatArray] = []
        for index, item in enumerate(self.times):
            array: FloatArray = np.asarray(item)
            if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
                raise TypeError(f"query times {index} must be numeric, got dtype {array.dtype}")
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"query times {index} must have a non-empty shape (num_queries,), got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"query times {index} contain NaN or infinite values.")
            if np.any(np.diff(array) < 0):
                raise ValueError(f"query times {index} must not decrease.")
            validated.append(array.astype(np.float64))
        object.__setattr__(self, "times", tuple(validated))

    @staticmethod
    def from_arrays(times: Sequence[FloatArray]) -> QueryTimes:
        """
        Build the query times from one array per sequence.

        Parameters
        ----------
        times : Sequence[FloatArray]
            Non-decreasing times of each sequence.

        Returns
        -------
        QueryTimes
            The query times.

        Raises
        ------
        TypeError
            If times are not numeric.
        ValueError
            If the times are invalid.
        """
        return QueryTimes(tuple(times))

    @property
    def num_sequences(self) -> int:
        """
        Get the number of sequences.

        Returns
        -------
        int
            The number of sequences.
        """
        return len(self.times)

    @property
    def lengths(self) -> IndexArray:
        """
        Get the number of query times of each sequence.

        Returns
        -------
        IndexArray
            The counts with shape (num_sequences,).
        """
        return np.array([item.size for item in self.times], dtype=np.intp)

    def pad(self) -> FloatArray:
        """
        Pad the times of every sequence to the longest, repeating each sequence's last time.

        Returns
        -------
        FloatArray
            The times with shape (num_sequences, max_queries).
        """
        lengths: IndexArray = self.lengths
        padded: FloatArray = np.empty((self.num_sequences, int(lengths.max())), dtype=np.float64)
        for index, item in enumerate(self.times):
            padded[index, : item.size] = item
            padded[index, item.size :] = item[-1]
        return padded
