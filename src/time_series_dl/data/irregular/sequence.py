from dataclasses import dataclass

import numpy as np

from ...array_types import BoolArray, FloatArray


@dataclass(frozen=True, eq=False)
class IrregularSequence:
    """
    One irregularly sampled multivariate sequence: observation times and the values observed at them.

    Attributes
    ----------
    times : FloatArray
        Strictly increasing observation times with shape (length,), as float64.
    values : FloatArray
        Values with shape (length, channels), as float64; NaN marks a channel not observed at a time, and every time
        has at least one observed channel.

    Raises
    ------
    TypeError
        If the times or values are not numeric.
    ValueError
        If the shapes are invalid, a time is not finite or not after the previous one, a value is infinite, or a time
        has no observed value.
    """

    times: FloatArray
    values: FloatArray

    def __post_init__(self) -> None:
        times: FloatArray = np.asarray(self.times)
        values: FloatArray = np.asarray(self.values)
        for name, array in (("times", times), ("values", values)):
            if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
                raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
        if times.ndim != 1 or times.size == 0:
            raise ValueError(f"times must have a non-empty shape (length,), got {times.shape}")
        if values.ndim != 2 or values.shape[0] != times.size or values.shape[1] == 0:
            raise ValueError(f"values must have shape ({times.size}, channels), got {values.shape}")
        if not np.all(np.isfinite(times)):
            raise ValueError("times contain NaN or infinite values.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing.")
        if np.any(np.isinf(values)):
            raise ValueError("values contain infinite values.")
        is_unobserved: BoolArray = np.all(np.isnan(values), axis=1)
        if np.any(is_unobserved):
            raise ValueError(
                f"every time needs an observed value; none at steps {np.flatnonzero(is_unobserved).tolist()}"
            )
        object.__setattr__(self, "times", times.astype(np.float64))
        object.__setattr__(self, "values", values.astype(np.float64))

    @property
    def length(self) -> int:
        """
        Get the number of observation times.

        Returns
        -------
        int
            The length.
        """
        return int(self.times.size)

    @property
    def channels(self) -> int:
        """
        Get the number of channels.

        Returns
        -------
        int
            The number of channels.
        """
        return int(self.values.shape[1])
