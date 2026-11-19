from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import SequenceEncoder
from .encoder_layer import TSTEncoderLayer

if TYPE_CHECKING:
    from .parameters import TSTParameters


class TSTSequenceEncoder(SequenceEncoder):
    """
    TST encoder: a linear projection of each step's values scaled by ``sqrt(model_size)``, a learnable position
    encoding, dropout, the encoder layers and a final activation.

    Parameters
    ----------
    parameters : TSTParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If input_length or input_channels is not positive.
    """

    def __init__(self, parameters: TSTParameters, input_length: int, input_channels: int) -> None:
        if input_length <= 0 or input_channels <= 0:
            raise ValueError(
                f"input_length and input_channels must be positive, got {input_length} and {input_channels}"
            )
        super().__init__()
        self._output_length: int = input_length
        self._output_size: int = parameters.model_size
        self.input_scale: float = math.sqrt(parameters.model_size)
        self.input_projection: nn.Linear = nn.Linear(input_channels, parameters.model_size)
        self.position: nn.Parameter = nn.Parameter(
            torch.empty(input_length, parameters.model_size).uniform_(-0.02, 0.02)
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.layers: nn.Sequential = nn.Sequential(*(TSTEncoderLayer(parameters) for _ in range(parameters.num_layers)))
        self.activation: nn.Module = parameters.activation.build()

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
        tokens: Tensor = self.input_projection(values) * self.input_scale + self.position
        encoded: Tensor = self.layers(self.dropout(tokens))
        activated: Tensor = self.activation(encoded)
        return activated
