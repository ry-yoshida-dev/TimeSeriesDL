from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional


class HeadwiseLayerNorm(nn.Module):
    """
    Layer normalization of each head's features separately (the group normalization of xLSTM), with a learned scale
    and no bias.

    Parameters
    ----------
    size : int
        Size of the feature vectors.
    num_heads : int
        Number of heads; it must divide the size.

    Raises
    ------
    ValueError
        If a size is not positive or num_heads does not divide size.
    """

    def __init__(self, size: int, num_heads: int) -> None:
        if size <= 0 or num_heads <= 0:
            raise ValueError(f"size and num_heads must be positive, got {size} and {num_heads}")
        if size % num_heads != 0:
            raise ValueError(f"num_heads {num_heads} must divide size {size}")
        super().__init__()
        self.num_heads: int = num_heads
        self.head_size: int = size // num_heads
        self.weight: nn.Parameter = nn.Parameter(torch.ones(size))

    def forward(self, values: Tensor) -> Tensor:
        """
        Normalize each head.

        Parameters
        ----------
        values : Tensor
            Features whose last dimension is ``size``.

        Returns
        -------
        Tensor
            Normalized features with the shape of ``values``.
        """
        heads: Tensor = values.reshape(*values.shape[:-1], self.num_heads, self.head_size)
        normalized: Tensor = functional.layer_norm(heads, (self.head_size,)).flatten(start_dim=-2)
        return normalized * self.weight

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
