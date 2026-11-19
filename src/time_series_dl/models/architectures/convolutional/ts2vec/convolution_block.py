from typing import cast

import torch.nn.functional as functional
from torch import Tensor, nn


class TS2VecConvolutionBlock(nn.Module):
    """
    Residual block of two dilated convolutions, each preceded by GELU, that keep the length of the sequence.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    output_channels : int
        Number of output channels.
    kernel_size : int
        Length of the kernels.
    dilation : int
        Dilation of both convolutions.
    is_final : bool
        Whether this is the last block, whose residual connection is always projected, as in the official code.
    """

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int, dilation: int, is_final: bool
    ) -> None:
        super().__init__()
        padding: int = ((kernel_size - 1) * dilation + 1) // 2
        self.first: nn.Conv1d = nn.Conv1d(
            input_channels, output_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.second: nn.Conv1d = nn.Conv1d(
            output_channels, output_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.projection: nn.Conv1d | None = (
            nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels or is_final else None
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Apply the block.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, input_channels, length).

        Returns
        -------
        Tensor
            Features with shape (batch, output_channels, length).
        """
        length: int = features.shape[-1]
        residual: Tensor = features if self.projection is None else self.projection(features)
        hidden: Tensor = self.first(functional.gelu(features))[..., :length]
        hidden = self.second(functional.gelu(hidden))[..., :length]
        return hidden + residual

    def __call__(self, features: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(features))
