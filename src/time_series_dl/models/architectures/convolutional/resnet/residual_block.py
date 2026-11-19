from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn

from ....components.layers.convolution_block import ConvolutionBlock


class ResNetResidualBlock(nn.Module):
    """
    Length-preserving convolutions with batch normalization whose sum with a shortcut passes through a ReLU.

    Every convolution but the last is followed by a ReLU. The shortcut is a batch-normalized 1x1 convolution when the
    number of channels changes, and the batch-normalized input otherwise.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_sizes : tuple[int, ...]
        Kernel length of each successive convolution.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        sizes: list[int] = [input_channels, *(output_channels for _ in kernel_sizes)]
        self.convolutions: nn.Sequential = nn.Sequential(
            *(
                ConvolutionBlock(sizes[position], sizes[position + 1], kernel_size, position < len(kernel_sizes) - 1)
                for position, kernel_size in enumerate(kernel_sizes)
            )
        )
        self.shortcut: nn.Module = (
            ConvolutionBlock(input_channels, output_channels, 1, is_activated=False)
            if input_channels != output_channels
            else nn.BatchNorm1d(output_channels)
        )

    def forward(self, values: Tensor) -> Tensor:
        """
        Apply the block.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, input_channels, length).

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, length).
        """
        features: Tensor = self.convolutions(values)
        shortcut: Tensor = self.shortcut(values)
        return functional.relu(features + shortcut)

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
