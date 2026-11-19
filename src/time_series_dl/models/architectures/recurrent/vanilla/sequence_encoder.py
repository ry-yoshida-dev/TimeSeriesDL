from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SequenceEncoder

if TYPE_CHECKING:
    from .parameters import RNNParameters


class RNNSequenceEncoder(SequenceEncoder):
    """
    Stacked recurrent layers whose hidden states are the features of each step.

    Parameters
    ----------
    parameters : RNNParameters
        The layer configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.
    """

    def __init__(self, parameters: RNNParameters, input_length: int, input_channels: int) -> None:
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.hidden_size * (2 if parameters.is_bidirectional else 1)
        self.recurrent: nn.LSTM | nn.GRU | nn.RNN = parameters.cell.build(
            input_channels,
            parameters.hidden_size,
            parameters.num_layers,
            parameters.dropout,
            parameters.is_bidirectional,
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
        features: Tensor = self.recurrent(values)[0]
        return features
