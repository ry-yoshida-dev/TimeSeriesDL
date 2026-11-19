import math
from abc import ABC, abstractmethod
from typing import cast

import torch
from torch import Tensor, nn


class SSMKernel(nn.Module, ABC):
    """
    Convolution kernel of a linear time-invariant state space model, one single-input single-output model per channel.

    Parameters
    ----------
    size : int
        Number of channels.
    min_step : float
        Smallest initial step size.
    max_step : float
        Largest initial step size; the initial step sizes are log-uniform between the two.

    Raises
    ------
    ValueError
        If size is not positive or the step sizes are not 0 < min_step <= max_step.
    """

    def __init__(self, size: int, min_step: float, max_step: float) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if not 0.0 < min_step <= max_step:
            raise ValueError(f"steps must satisfy 0 < min_step <= max_step, got {min_step} and {max_step}")
        super().__init__()
        self.log_step: nn.Parameter = nn.Parameter(
            torch.rand(size) * (math.log(max_step) - math.log(min_step)) + math.log(min_step)
        )

    @abstractmethod
    def forward(self, length: int) -> Tensor:
        """
        Compute the kernel.

        Parameters
        ----------
        length : int
            Number of lags.

        Returns
        -------
        Tensor
            The real kernel with shape (size, length).
        """

    def __call__(self, length: int) -> Tensor:
        return cast(Tensor, super().__call__(length))
