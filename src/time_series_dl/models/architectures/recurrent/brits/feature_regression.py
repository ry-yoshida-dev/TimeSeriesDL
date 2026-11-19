from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class BRITSFeatureRegression(nn.Module):
    """
    Estimates each channel from the other channels of the same time step: a linear map whose diagonal is fixed at zero
    (Cao et al., 2018).

    Parameters
    ----------
    channels : int
        Number of channels.

    Raises
    ------
    ValueError
        If channels is not positive.
    """

    def __init__(self, channels: int) -> None:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        super().__init__()
        bound: float = channels**-0.5
        self.weight: nn.Parameter = nn.Parameter(torch.empty(channels, channels).uniform_(-bound, bound))
        self.bias: nn.Parameter = nn.Parameter(torch.empty(channels).uniform_(-bound, bound))
        self.off_diagonal: Tensor
        self.register_buffer("off_diagonal", 1.0 - torch.eye(channels), persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        """
        Estimate every channel from the others.

        Parameters
        ----------
        values : Tensor
            Values with shape (..., channels).

        Returns
        -------
        Tensor
            Estimates of the shape of ``values``.
        """
        return functional.linear(values, self.weight * self.off_diagonal, self.bias)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
