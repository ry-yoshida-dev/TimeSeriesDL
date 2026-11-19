from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.layers.same_length_convolution import SameLengthConvolution


class InceptionTimeModule(nn.Module):
    """
    Parallel convolutions of several lengths and a max-pooling branch, concatenated, batch-normalized and activated.

    The convolution branches read a 1x1 bottleneck of the input when the input has more than one channel.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    num_filters : int
        Number of filters of each branch.
    bottleneck_size : int
        Number of channels of the bottleneck.
    kernel_sizes : tuple[int, ...]
        Kernel length of each convolution branch.
    """

    def __init__(
        self, input_channels: int, num_filters: int, bottleneck_size: int, kernel_sizes: tuple[int, ...]
    ) -> None:
        super().__init__()
        is_bottlenecked: bool = input_channels > 1
        branch_input: int = bottleneck_size if is_bottlenecked else input_channels
        self.bottleneck: nn.Module = (
            SameLengthConvolution(input_channels, bottleneck_size, 1, is_biased=False)
            if is_bottlenecked
            else nn.Identity()
        )
        self.branches: nn.ModuleList = nn.ModuleList(
            SameLengthConvolution(branch_input, num_filters, kernel_size, is_biased=False)
            for kernel_size in kernel_sizes
        )
        self.pooling: nn.MaxPool1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pooled_convolution: SameLengthConvolution = SameLengthConvolution(
            input_channels, num_filters, 1, is_biased=False
        )
        self.normalization: nn.BatchNorm1d = nn.BatchNorm1d(num_filters * (len(kernel_sizes) + 1))

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
            Features with shape (batch, num_filters * (len(kernel_sizes) + 1), length).
        """
        reduced: Tensor = self.bottleneck(values)
        outputs: list[Tensor] = [branch(reduced) for branch in self.branches]
        outputs.append(self.pooled_convolution(self.pooling(values)))
        features: Tensor = self.normalization(torch.cat(outputs, dim=1))
        return functional.relu(features)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
