import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional


class CausalDepthwiseConvolution(nn.Module):
    """
    Causal convolution of each channel with its own short kernel, along the time axis of (batch, length, channels)
    values, as in the Mamba blocks.

    It is computed from shifted copies of the input, which is much faster on the CPU than a grouped ``nn.Conv1d``.

    Parameters
    ----------
    channels : int
        Number of channels.
    kernel_size : int
        Number of steps each output reads, the current one included.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, channels: int, kernel_size: int) -> None:
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        super().__init__()
        self.kernel_size: int = kernel_size
        bound: float = 1.0 / math.sqrt(kernel_size)
        self.weight: nn.Parameter = nn.Parameter(torch.empty(channels, kernel_size).uniform_(-bound, bound))
        self.bias: nn.Parameter = nn.Parameter(torch.empty(channels).uniform_(-bound, bound))

    def forward(self, values: Tensor) -> Tensor:
        """
        Convolve causally along time.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels).

        Returns
        -------
        Tensor
            Outputs with the shape of ``values``.
        """
        padded: Tensor = functional.pad(values, (0, 0, self.kernel_size - 1, 0))
        windows: Tensor = padded.unfold(1, self.kernel_size, 1)
        convolved: Tensor = torch.einsum("blck,ck->blc", windows, self.weight) + self.bias
        return convolved

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
