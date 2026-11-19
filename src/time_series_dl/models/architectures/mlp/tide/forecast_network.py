from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .residual_block import TiDEResidualBlock

if TYPE_CHECKING:
    from .parameters import TiDEParameters


class TiDEForecastNetwork(ForecastNetwork):
    """
    Channel-independent TiDE without covariates.

    The forecast of each target channel reads only that channel's past values: its window is encoded and decoded by
    dense residual blocks into one vector per horizon step, each vector is mapped to the distribution parameters by
    the temporal decoder, and a linear map of the window to the horizon is added. All weights are shared by the
    channels. The temporal decoder is never layer-normalized, since its outputs are the distribution parameters.

    Parameters
    ----------
    parameters : TiDEParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: TiDEParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.decoder_output_size: int = parameters.decoder_output_size
        self.encoder: nn.Sequential = parameters.build_dense_stack(
            shape.input_length, parameters.hidden_size, parameters.num_encoder_layers
        )
        self.decoder: nn.Sequential = parameters.build_dense_stack(
            parameters.hidden_size, shape.horizon * parameters.decoder_output_size, parameters.num_decoder_layers
        )
        self.temporal_decoder: TiDEResidualBlock = TiDEResidualBlock(
            parameters.decoder_output_size,
            parameters.temporal_decoder_hidden_size,
            shape.num_parameters,
            parameters.dropout,
            is_layer_normalized=False,
        )
        self.residual: nn.Linear = nn.Linear(shape.input_length, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2)
        encoding: Tensor = self.encoder(series)
        decoded: Tensor = self.decoder(encoding)
        steps: Tensor = decoded.reshape(*series.shape[:2], self.horizon, self.decoder_output_size)
        temporal: Tensor = self.temporal_decoder(steps)
        residual: Tensor = self.residual(series).reshape(*series.shape[:2], self.horizon, self.num_parameters)
        return (temporal + residual).transpose(1, 2)
