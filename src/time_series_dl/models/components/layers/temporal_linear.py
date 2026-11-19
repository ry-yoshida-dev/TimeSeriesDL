import math
from typing import cast

import torch
from torch import Tensor, nn


class TemporalLinear(nn.Module):
    """
    Linear map along the time axis, applied to every channel.

    Parameters
    ----------
    input_length : int
        Number of input time steps.
    output_length : int
        Number of output time steps.
    channels : int
        Number of channels.
    is_individual : bool
        Whether each channel has its own weights instead of sharing one map.
    """

    def __init__(self, input_length: int, output_length: int, channels: int, is_individual: bool) -> None:
        super().__init__()
        count: int = channels if is_individual else 1
        bound: float = 1.0 / math.sqrt(input_length)
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty(count, input_length, output_length).uniform_(-bound, bound)
        )
        self.bias: nn.Parameter = nn.Parameter(torch.empty(count, output_length).uniform_(-bound, bound))

    def forward(self, values: Tensor) -> Tensor:
        """
        Map a batch of sequences along time.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_length, channels).

        Returns
        -------
        Tensor
            Values with shape (batch, output_length, channels).
        """
        mapped: Tensor = torch.einsum("blc,clm->bmc", values, self.weight.expand(values.shape[2], -1, -1))
        return mapped + self.bias.expand(values.shape[2], -1).transpose(0, 1).unsqueeze(0)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
