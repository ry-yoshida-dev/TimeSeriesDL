from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SequenceEncoder
from .residual_block import TemporalResidualBlock

if TYPE_CHECKING:
    from .parameters import TCNParameters


class TCNSequenceEncoder(SequenceEncoder):
    """
    Stack of residual blocks whose dilation doubles at each level.

    Parameters
    ----------
    parameters : TCNParameters
        The block configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: TCNParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.channel_sizes[-1]
        sizes: list[int] = [input_channels, *parameters.channel_sizes]
        self.blocks: nn.Sequential = nn.Sequential(
            *(
                TemporalResidualBlock(
                    sizes[level], sizes[level + 1], parameters.kernel_size, 2**level, parameters.dropout
                )
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
