from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .damping_layer import ETSformerDampingLayer
from .encoder import ETSformerEncoder
from .encoding import ETSformerEncoding

if TYPE_CHECKING:
    from .parameters import ETSformerParameters


class ETSformerForecastNetwork(ForecastNetwork):
    """
    ETSformer encoder-decoder: the decoder damps the last growth of each layer over the horizon and takes the
    extrapolated seasonality of each layer; their sums are mapped linearly to the parameters of the target channels,
    and the last level of each target channel is added to the parameters measured in the scale of the values.

    Parameters
    ----------
    parameters : ETSformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: ETSformerParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.encoder: ETSformerEncoder = ETSformerEncoder(
            parameters, shape.input_channels, shape.horizon, is_level_tracked=True
        )
        self.damping_layers: nn.ModuleList = nn.ModuleList(
            ETSformerDampingLayer(shape.horizon, parameters.num_heads, parameters.dropout)
            for _ in range(parameters.num_layers)
        )
        self.growth_dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.output_channels * shape.num_parameters)
        level_mask: Tensor = torch.zeros(shape.num_parameters)
        level_mask[list(shape.level_parameters)] = 1.0
        self.level_mask: Tensor
        self.register_buffer("level_mask", level_mask, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        encoding: ETSformerEncoding = self.encoder(past)
        growth: Tensor = past.new_zeros(())
        for damping_layer, layer_growth in zip(self.damping_layers, encoding.growths, strict=True):
            damped: Tensor = damping_layer(layer_growth[:, -1:])
            growth = growth + self.growth_dropout(damped)
        season: Tensor = torch.stack([layer_season[:, -self.horizon :] for layer_season in encoding.seasons]).sum(0)
        outputs: Tensor = self.projection(growth + season).reshape(
            -1, self.horizon, self.output_channels, self.num_parameters
        )
        if encoding.level is None:
            return outputs
        last_level: Tensor = encoding.level[:, -1:].index_select(2, self.target_channels)
        return outputs + last_level.unsqueeze(-1) * self.level_mask
