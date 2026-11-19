from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....components.layers.convolution_block import ConvolutionBlock
from ....interfaces import SequenceEncoder

if TYPE_CHECKING:
    from .parameters import FCNParameters


class FCNSequenceEncoder(SequenceEncoder):
    """
    Stack of length-preserving convolution blocks, each with batch normalization and a ReLU.

    Parameters
    ----------
    parameters : FCNParameters
        The block configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: FCNParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.channel_sizes[-1]
        sizes: list[int] = [input_channels, *parameters.channel_sizes]
        self.blocks: nn.Sequential = nn.Sequential(
            *(
                ConvolutionBlock(sizes[level], sizes[level + 1], kernel_size)
                for level, kernel_size in enumerate(parameters.kernel_sizes)
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
