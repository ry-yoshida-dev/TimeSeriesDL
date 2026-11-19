from typing import cast

import torch
from torch import Tensor, nn


class TTMGatedAttention(nn.Module):
    """
    Gated attention of TinyTimeMixer: the features are multiplied by a softmax over a linear projection of themselves.

    Parameters
    ----------
    size : int
        Size of the last dimension.
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.projection: nn.Linear = nn.Linear(size, size)

    def forward(self, values: Tensor) -> Tensor:
        """
        Gate the last dimension.

        Parameters
        ----------
        values : Tensor
            Values with shape (..., size).

        Returns
        -------
        Tensor
            The gated values with the same shape.
        """
        return values * torch.softmax(self.projection(values), dim=-1)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
