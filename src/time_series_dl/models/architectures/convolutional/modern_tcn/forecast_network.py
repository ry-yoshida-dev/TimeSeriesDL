from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import ModernTCNEncoder

if TYPE_CHECKING:
    from .parameters import ModernTCNParameters


class ModernTCNForecastNetwork(ForecastNetwork):
    """
    ModernTCN backbone over every input variable, with each target variable's flattened features mapped linearly to
    its horizon by a head shared across variables.

    Parameters
    ----------
    parameters : ModernTCNParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: ModernTCNParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.encoder: ModernTCNEncoder = ModernTCNEncoder(parameters, shape.input_length, shape.input_channels)
        self.dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.projection: nn.Linear = nn.Linear(
            self.encoder.output_size * self.encoder.output_length, shape.horizon * shape.num_parameters
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        features: Tensor = self.encoder(past).index_select(1, self.target_channels)
        outputs: Tensor = self.projection(self.dropout(features.flatten(start_dim=2)))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
