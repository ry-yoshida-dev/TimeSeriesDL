import numpy as np
from numpy.typing import NDArray

from ...array_types import FloatArray


class CovariateArrayValidation:
    """
    Checks shared by the covariate containers: covariate arrays must be finite, non-empty and of a fixed number of
    dimensions, and categorical codes must be non-negative integers.
    """

    @staticmethod
    def require_real(values: FloatArray, name: str, num_dimensions: int) -> FloatArray:
        """
        Validate real-valued covariates.

        Parameters
        ----------
        values : FloatArray
            The covariates.
        name : str
            The name of the covariates in error messages.
        num_dimensions : int
            The number of dimensions they must have.

        Returns
        -------
        FloatArray
            The covariates as float32.

        Raises
        ------
        TypeError
            If the values are not numeric.
        ValueError
            If the array has another number of dimensions, is empty, or contains NaN or infinite values.
        """
        array: FloatArray = np.asarray(values)
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
        if array.ndim != num_dimensions or 0 in array.shape:
            raise ValueError(f"{name} must have {num_dimensions} non-empty dimensions, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return array.astype(np.float32)

    @staticmethod
    def require_codes(values: NDArray[np.integer], name: str, num_dimensions: int) -> NDArray[np.int64]:
        """
        Validate integer codes of categorical covariates.

        Parameters
        ----------
        values : NDArray[np.integer]
            The codes.
        name : str
            The name of the codes in error messages.
        num_dimensions : int
            The number of dimensions they must have.

        Returns
        -------
        NDArray[np.int64]
            The codes as int64.

        Raises
        ------
        TypeError
            If the codes are not integers.
        ValueError
            If the array has another number of dimensions, is empty, or contains a negative code.
        """
        array: NDArray[np.integer] = np.asarray(values)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"{name} must be integer codes, got dtype {array.dtype}")
        if array.ndim != num_dimensions or 0 in array.shape:
            raise ValueError(f"{name} must have {num_dimensions} non-empty dimensions, got shape {array.shape}")
        if np.any(array < 0):
            raise ValueError(f"{name} must not contain negative codes.")
        return array.astype(np.int64)
