from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SequenceEncoder
from .residual_block import ResNetResidualBlock

if TYPE_CHECKING:
    from .parameters import ResNetParameters


class ResNetSequenceEncoder(SequenceEncoder):
    """
    Stack of length-preserving residual convolution blocks.

    Parameters
    ----------
    parameters : ResNetParameters
        The block configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: ResNetParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.channel_sizes[-1]
        sizes: list[int] = [input_channels, *parameters.channel_sizes]
        self.blocks: nn.Sequential = nn.Sequential(
            *(
                ResNetResidualBlock(sizes[level], sizes[level + 1], parameters.kernel_sizes)
                for level in range(len(parameters.channel_sizes))
            )
        )

    @property
    @override
    def output_length(self) -> int:
        return self._output_length

    @property
    @override
    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.blocks(values.transpose(1, 2))
        return features.transpose(1, 2)
