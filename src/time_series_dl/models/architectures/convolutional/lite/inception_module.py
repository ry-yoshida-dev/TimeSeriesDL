from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.layers.same_length_convolution import SameLengthConvolution
from .custom_filters import LITECustomFilters


class LITEInceptionModule(nn.Module):
    """
    First module of LITE: parallel convolutions of halving lengths (multiplexing), optionally with the fixed custom
    filters, concatenated, batch-normalized and activated by a ReLU.

    Without multiplexing a single convolution of the longest length has as many filters as the branches together.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    num_filters : int
        Number of filters of each branch.
    kernel_size : int
        Length of the longest branch; branch i has length ``kernel_size // 2**i``.
    is_multiplexed : bool
        Whether three branches of halving lengths are used instead of one.
    is_custom_filtered : bool
        Whether the custom filters are concatenated to the branches.
    """

    def __init__(
        self,
        input_channels: int,
        num_filters: int,
        kernel_size: int,
        is_multiplexed: bool,
        is_custom_filtered: bool,
    ) -> None:
        super().__init__()
        num_branches: int = 3 if is_multiplexed else 1
        branch_filters: int = num_filters if is_multiplexed else 3 * num_filters
        self.branches: nn.ModuleList = nn.ModuleList(
            SameLengthConvolution(input_channels, branch_filters, kernel_size // 2**branch, is_biased=False)
            for branch in range(num_branches)
        )
        self.custom_filters: LITECustomFilters | None = (
            LITECustomFilters(input_channels) if is_custom_filtered else None
        )
        num_custom: int = 0 if self.custom_filters is None else self.custom_filters.num_filters
        self.output_size: int = 3 * num_filters + num_custom
        self.normalization: nn.BatchNorm1d = nn.BatchNorm1d(self.output_size)

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the module.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_channels, length).

        Returns
        -------
        Tensor
            Features with shape (batch, output_size, length).
        """
        outputs: list[Tensor] = [branch(values) for branch in self.branches]
        if self.custom_filters is not None:
            outputs.append(self.custom_filters(values))
        return functional.relu(self.normalization(torch.cat(outputs, dim=1)))

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
