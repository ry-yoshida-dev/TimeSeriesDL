from typing import cast

import torch
from torch import Tensor, nn


class T5LayerNorm(nn.Module):
    """
    Root-mean-square layer normalization of T5: scaling by the root mean square without centering or bias, computed
    in float32.

    Parameters
    ----------
    size : int
        Size of the normalized last dimension.
    epsilon : float
        Added to the mean square.
    """

    def __init__(self, size: int, epsilon: float) -> None:
        super().__init__()
        self.epsilon: float = epsilon
        self.weight: nn.Parameter = nn.Parameter(torch.ones(size))

    def forward(self, values: Tensor) -> Tensor:
        """
        Normalize the last dimension.

        Parameters
        ----------
        values : Tensor
            Values with shape (..., size).

        Returns
        -------
        Tensor
            The normalized values in the dtype of the weight.
        """
        mean_square: Tensor = values.to(torch.float32).square().mean(dim=-1, keepdim=True)
        normalized: Tensor = values * torch.rsqrt(mean_square + self.epsilon)
        return self.weight * normalized.to(self.weight.dtype)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
