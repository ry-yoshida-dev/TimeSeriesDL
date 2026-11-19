from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .encoder_layer import TimeXerEncoderLayer
from .endogenous_embedding import TimeXerEndogenousEmbedding

if TYPE_CHECKING:
    from .parameters import TimeXerParameters


class TimeXerForecastNetwork(ForecastNetwork):
    """
    TimeXer: the target channels are the endogenous variates, embedded as patch tokens and a global token each, and
    the other input channels are the exogenous variates, each embedded as one token from its whole series. When every
    input channel is a target, every channel is also an exogenous variate, as in the paper's multivariate setting. The
    encoded tokens of each endogenous variate are flattened and mapped linearly to its horizon.

    Parameters
    ----------
    parameters : TimeXerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: TimeXerParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        exogenous_channels: tuple[int, ...] = tuple(
            channel for channel in range(shape.input_channels) if channel not in shape.target_channels
        ) or tuple(range(shape.input_channels))
        self.endogenous_channels: Tensor
        self.register_buffer("endogenous_channels", torch.tensor(shape.target_channels), persistent=False)
        self.exogenous_channels: Tensor
        self.register_buffer("exogenous_channels", torch.tensor(exogenous_channels), persistent=False)
        self.endogenous_embedding: TimeXerEndogenousEmbedding = TimeXerEndogenousEmbedding(
            parameters, shape.input_length, shape.output_channels
        )
        self.exogenous_embedding: nn.Linear = nn.Linear(shape.input_length, parameters.model_size)
        self.exogenous_dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.layers: nn.ModuleList = nn.ModuleList(
            TimeXerEncoderLayer(parameters) for _ in range(parameters.num_layers)
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(parameters.model_size)
        self.projection: nn.Linear = nn.Linear(
            parameters.model_size * (self.endogenous_embedding.num_patches + 1), shape.horizon * shape.num_parameters
        )
        self.head_dropout: nn.Dropout = nn.Dropout(parameters.dropout)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch: int = past.shape[0]
        series: Tensor = past.transpose(1, 2)
        tokens: Tensor = self.endogenous_embedding(series.index_select(1, self.endogenous_channels))
        exogenous: Tensor = self.exogenous_dropout(
            self.exogenous_embedding(series.index_select(1, self.exogenous_channels))
        )
        for layer in self.layers:
            tokens = cast(TimeXerEncoderLayer, layer)(tokens, exogenous)
        encoded: Tensor = self.norm(tokens).reshape(batch, self.output_channels, -1)
        outputs: Tensor = self.head_dropout(self.projection(encoded))
        return outputs.reshape(batch, self.output_channels, self.horizon, self.num_parameters).transpose(1, 2)
