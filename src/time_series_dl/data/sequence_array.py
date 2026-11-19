from dataclasses import dataclass

import numpy as np

from ..array_types import FloatArray


@dataclass(frozen=True, eq=False)
class SequenceArray:
    """
    Validated batch of equally long multivariate sequences.

    Attributes
    ----------
    values : FloatArray
        Sequences with shape (num_sequences, length, channels), as float32.
    is_nan_allowed : bool
        Whether NaN marks missing values; infinite values are always rejected.

    Raises
    ------
    TypeError
        If the values are not numeric.
    ValueError
        If the array is not 3-D, is empty, or contains values that are not allowed.
    """

    values: FloatArray
    is_nan_allowed: bool = False

    def __post_init__(self) -> None:
        array: FloatArray = np.asarray(self.values)
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise TypeError(f"sequences must be numeric, got dtype {array.dtype}")
        if array.ndim != 3 or 0 in array.shape:
            raise ValueError(
                f"sequences must have a non-empty shape (num_sequences, length, channels), got {array.shape}"
            )
        if np.any(np.isinf(array)):
            raise ValueError("sequences contain infinite values.")
        if not self.is_nan_allowed and np.any(np.isnan(array)):
            raise ValueError("sequences contain NaN values.")
        object.__setattr__(self, "values", array.astype(np.float32))

    @property
    def num_sequences(self) -> int:
        """
        Get the number of sequences.

        Returns
        -------
        int
            The number of sequences.
        """
        return int(self.values.shape[0])

    @property
    def length(self) -> int:
        """
        Get the number of time steps per sequence.

        Returns
        -------
        int
            The length.
        """
        return int(self.values.shape[1])

    @property
    def channels(self) -> int:
        """
        Get the number of channels.

        Returns
        -------
        int
            The number of channels.
        """
        return int(self.values.shape[2])

    def require_shape(self, length: int, channels: int) -> None:
        """
        Check that the sequences have the dimensions a fitted network expects.

        Parameters
        ----------
        length : int
            The expected number of time steps.
        channels : int
            The expected number of channels.

        Raises
        ------
        ValueError
            If the length or the number of channels differs.
        """
        if (self.length, self.channels) != (length, channels):
            raise ValueError(
                f"sequences must have length {length} and {channels} channels, got {self.length} and {self.channels}"
            )
