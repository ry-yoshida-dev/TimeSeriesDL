from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SequenceEncoder
from .inception_module import LITEInceptionModule
from .separable_convolution_block import LITESeparableConvolutionBlock

if TYPE_CHECKING:
    from .parameters import LITEParameters

_NUM_SEPARABLE_BLOCKS: int = 2


class LITESequenceEncoder(SequenceEncoder):
    """
    LITE network without its output layer: an Inception module followed by two depthwise separable convolution
    blocks, every layer keeping the length.

    The separable blocks have kernel lengths ``kernel_size // 2`` and ``kernel_size // 4`` and, with dilation,
    dilations 2 and 4, as in the official code.

    Parameters
    ----------
    parameters : LITEParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: LITEParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.num_filters
        self.inception: LITEInceptionModule = LITEInceptionModule(
            input_channels,
            parameters.num_filters,
            parameters.kernel_size,
            parameters.is_multiplexed,
            parameters.is_custom_filtered,
        )
        block_kernel_size: int = parameters.kernel_size // 2
        self.blocks: nn.Sequential = nn.Sequential(
            *(
                LITESeparableConvolutionBlock(
                    self.inception.output_size if position == 0 else parameters.num_filters,
                    parameters.num_filters,
                    block_kernel_size // 2**position,
                    2 ** (position + 1) if parameters.is_dilated else 1,
                )
                for position in range(_NUM_SEPARABLE_BLOCKS)
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
        features: Tensor = self.blocks(self.inception(values.transpose(1, 2)))
        return features.transpose(1, 2)
