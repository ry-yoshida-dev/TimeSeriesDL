from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape

if TYPE_CHECKING:
    from .parameters import SegRNNParameters


class SegRNNForecastNetwork(ForecastNetwork):
    """
    Channel-independent segment recurrent network with parallel multi-step decoding.

    The forecast of each target channel reads only that channel's past values. The window is padded at its start by
    repeating its first value up to a whole number of segments; each segment is embedded linearly and the embeddings
    are encoded by a GRU. For every output segment, one GRU step from the final state reads the embedding of the
    segment's position (and of the target channel), and a linear layer maps the result to the segment's distribution
    parameters. The horizon is rounded up to whole segments and the excess steps are dropped. All weights but the
    channel embeddings are shared by the channels.

    Parameters
    ----------
    parameters : SegRNNParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.

    Raises
    ------
    ValueError
        If the segment is longer than the input window.
    """

    def __init__(self, parameters: SegRNNParameters, shape: ForecastShape) -> None:
        if parameters.segment_length > shape.input_length:
            raise ValueError(
                f"segment_length {parameters.segment_length} must not exceed input_length {shape.input_length}"
            )
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.segment_length: int = parameters.segment_length
        num_input_segments: int = math.ceil(shape.input_length / parameters.segment_length)
        self.input_padding: int = num_input_segments * parameters.segment_length - shape.input_length
        self.num_output_segments: int = math.ceil(shape.horizon / parameters.segment_length)
        self.segment_embedding: nn.Linear = nn.Linear(parameters.segment_length, parameters.hidden_size)
        self.recurrent: nn.GRU = nn.GRU(parameters.hidden_size, parameters.hidden_size, batch_first=True)
        position_size: int = parameters.hidden_size // 2 if parameters.is_channel_embedded else parameters.hidden_size
        self.position_embedding: nn.Parameter = nn.Parameter(torch.randn(self.num_output_segments, position_size))
        self.channel_embedding: nn.Parameter | None = (
            nn.Parameter(torch.randn(shape.output_channels, parameters.hidden_size // 2))
            if parameters.is_channel_embedded
            else None
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.hidden_size, parameters.segment_length * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    def _decoder_inputs(self, num_channels: int) -> Tensor:
        positions: Tensor = self.position_embedding.unsqueeze(0).expand(num_channels, -1, -1)
        if self.channel_embedding is None:
            return positions
        channels: Tensor = self.channel_embedding.unsqueeze(1).expand(-1, self.num_output_segments, -1)
        return torch.cat([positions, channels], dim=-1)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch_size: int = past.shape[0]
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2)
        num_channels: int = series.shape[1]
        if self.input_padding > 0:
            series = torch.cat([series[..., :1].expand(-1, -1, self.input_padding), series], dim=-1)
        segments: Tensor = series.reshape(batch_size * num_channels, -1, self.segment_length)
        embedded: Tensor = functional.relu(self.segment_embedding(segments))
        state: Tensor = self.recurrent(embedded)[1]
        decoder_state: Tensor = state.repeat_interleave(self.num_output_segments, dim=1)
        decoder_inputs: Tensor = (
            self._decoder_inputs(num_channels).repeat(batch_size, 1, 1).reshape(-1, 1, decoder_state.shape[-1])
        )
        decoded: Tensor = self.recurrent(decoder_inputs, decoder_state)[0]
        projected: Tensor = self.projection(self.dropout(decoded))
        outputs: Tensor = projected.reshape(
            batch_size, num_channels, self.num_output_segments * self.segment_length, self.num_parameters
        )
        return outputs[:, :, : self.horizon].transpose(1, 2)
