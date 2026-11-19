from typing import cast

import torch
from torch import Tensor, nn


class SeriesDecomposition(nn.Module):
    """
    Splits a series into a trend, its moving average, and the seasonal remainder.

    The series is padded by repeating its first and last values, so the trend has the length of the series.

    Parameters
    ----------
    kernel_size : int
        Length of the moving-average window; odd, so the window is centered.

    Raises
    ------
    ValueError
        If kernel_size is not a positive odd number.
    """

    def __init__(self, kernel_size: int) -> None:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd number, got {kernel_size}")
        super().__init__()
        self.kernel_size: int = kernel_size
        self.average: nn.AvgPool1d = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """
        Decompose a batch of series.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        tuple[Tensor, Tensor]
            The seasonal part and the trend, each with the shape of ``values``.
        """
        padding: int = (self.kernel_size - 1) // 2
        front: Tensor = values[:, :1, :].expand(-1, padding, -1)
        back: Tensor = values[:, -1:, :].expand(-1, padding, -1)
        padded: Tensor = torch.cat([front, values, back], dim=1)
        trend: Tensor = self.average(padded.transpose(1, 2)).transpose(1, 2)
        return values - trend, trend

    def __call__(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values))
