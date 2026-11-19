from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from ....interfaces import SequenceEncoder

if TYPE_CHECKING:
    from .parameters import LSTNetParameters


class LSTNetSequenceEncoder(SequenceEncoder):
    """
    Convolutional and recurrent components of LSTNet, which encode a window into one feature vector.

    A convolution over time and all channels, with ReLU, extracts short-term patterns. A GRU reads the convolved
    sequence and its last hidden state is kept. The recurrent-skip GRU reads, for each phase of the period, the
    convolved steps one period apart over the last whole periods; its last hidden states of every phase are
    concatenated to the GRU's. Dropout follows the convolution and both recurrent outputs.

    Parameters
    ----------
    parameters : LSTNetParameters
        The model configuration.
    input_length : int
        Number of time steps of the input.
    input_channels : int
        Number of channels of the input.

    Raises
    ------
    ValueError
        If the kernel is longer than the input or the convolved sequence holds no whole period of the skip component.
    """

    def __init__(self, parameters: LSTNetParameters, input_length: int, input_channels: int) -> None:
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if parameters.convolution_size > input_length:
            raise ValueError(
                f"convolution_size {parameters.convolution_size} must not exceed input_length {input_length}"
            )
        convolved_length: int = input_length - parameters.convolution_size + 1
        self.skip_period: int = parameters.skip_period
        self.num_periods: int = convolved_length // parameters.skip_period if parameters.skip_period > 0 else 0
        if parameters.skip_period > 0 and self.num_periods == 0:
            raise ValueError(
                f"skip_period {parameters.skip_period} must not exceed the convolved length {convolved_length}"
            )
        super().__init__()
        self.convolution: nn.Conv1d = nn.Conv1d(
            input_channels, parameters.convolution_channels, parameters.convolution_size
        )
        self.recurrent: nn.GRU = nn.GRU(parameters.convolution_channels, parameters.recurrent_size, batch_first=True)
        self.skip_recurrent: nn.GRU | None = (
            nn.GRU(parameters.convolution_channels, parameters.skip_recurrent_size, batch_first=True)
            if parameters.skip_period > 0
            else None
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self._output_size: int = parameters.recurrent_size + parameters.skip_period * parameters.skip_recurrent_size

    @property
    @override
    def output_length(self) -> int:
        return 1

    @property
    @override
    def output_size(self) -> int:
        return self._output_size

    @override
    def forward(self, values: Tensor) -> Tensor:
        batch_size: int = values.shape[0]
        convolved: Tensor = self.dropout(functional.relu(self.convolution(values.transpose(1, 2)))).transpose(1, 2)
        features: Tensor = self.dropout(self.recurrent(convolved)[1][0])
        if self.skip_recurrent is not None:
            periods: Tensor = convolved[:, convolved.shape[1] - self.num_periods * self.skip_period :]
            phases: Tensor = (
                periods.reshape(batch_size, self.num_periods, self.skip_period, -1)
                .transpose(1, 2)
                .reshape(batch_size * self.skip_period, self.num_periods, -1)
            )
            skip_features: Tensor = self.skip_recurrent(phases)[1][0].reshape(batch_size, -1)
            features = torch.cat([features, self.dropout(skip_features)], dim=-1)
        return features.unsqueeze(1)
