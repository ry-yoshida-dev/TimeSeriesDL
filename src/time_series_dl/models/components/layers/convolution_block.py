from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn

from .same_length_convolution import SameLengthConvolution


class ConvolutionBlock(nn.Module):
    """
    Length-preserving convolution followed by batch normalization and, optionally, a ReLU.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_size : int
        Length of the kernel.
    is_activated : bool
        Whether a ReLU follows the normalization.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, is_activated: bool = True) -> None:
        super().__init__()
        self.convolution: SameLengthConvolution = SameLengthConvolution(
            input_channels, output_channels, kernel_size, is_biased=False
        )
        self.normalization: nn.BatchNorm1d = nn.BatchNorm1d(output_channels)
        self.is_activated: bool = is_activated

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
        features: Tensor = self.normalization(self.convolution(values))
        return functional.relu(features) if self.is_activated else features

    def __call__(self, values: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(values))
