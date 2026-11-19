from enum import Enum

import torch
import torch.nn.functional as functional
from torch import Tensor


class NHITSInterpolation(Enum):
    """
    Interpolation of the coarse forecast coefficients of an N-HiTS block to every step of the horizon.

    Attributes
    ----------
    LINEAR
        Linear interpolation, the paper's default.
    NEAREST
        Each step takes the value of its nearest coefficient.
    """

    LINEAR = "linear"
    NEAREST = "nearest"

    def build_basis(self, num_coefficients: int, horizon: int) -> Tensor:
        """
        Build the interpolation as a linear basis.

        Parameters
        ----------
        num_coefficients : int
            Number of coarse coefficients.
        horizon : int
            Number of forecast steps.

        Returns
        -------
        Tensor
            The basis with shape (num_coefficients, horizon); row k is the interpolation of the k-th unit vector.

        Raises
        ------
        ValueError
            If a size is not positive.
        """
        if num_coefficients <= 0 or horizon <= 0:
            raise ValueError(f"num_coefficients and horizon must be positive, got {num_coefficients} and {horizon}")
        units: Tensor = torch.eye(num_coefficients).unsqueeze(1)
        match self:
            case NHITSInterpolation.LINEAR:
                return functional.interpolate(units, size=horizon, mode="linear").squeeze(1)
            case NHITSInterpolation.NEAREST:
                return functional.interpolate(units, size=horizon, mode="nearest").squeeze(1)
