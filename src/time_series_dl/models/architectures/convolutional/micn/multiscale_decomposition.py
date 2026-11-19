from typing import cast

import torch
from torch import Tensor, nn

from ....components.layers import SeriesDecomposition


class MICNMultiScaleDecomposition(nn.Module):
    """
    Multi-scale hybrid decomposition of MICN: the trend is the mean of moving averages of several lengths, and the
    seasonal part is the remainder.

    Parameters
    ----------
    kernel_sizes : tuple[int, ...]
        Odd length of each moving-average window.
    """

    def __init__(self, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.decompositions: nn.ModuleList = nn.ModuleList(SeriesDecomposition(size) for size in kernel_sizes)

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
        trend: Tensor = torch.stack([decomposition(values)[1] for decomposition in self.decompositions]).mean(dim=0)
        return values - trend, trend

    def __call__(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values))
