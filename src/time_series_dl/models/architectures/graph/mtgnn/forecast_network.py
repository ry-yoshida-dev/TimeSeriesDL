from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import MTGNNEncoder

if TYPE_CHECKING:
    from .parameters import MTGNNParameters


class MTGNNForecastNetwork(ForecastNetwork):
    """
    MTGNN over every input channel as a node, with a pointwise output layer that maps each target node's features to
    its whole horizon at once.

    Parameters
    ----------
    parameters : MTGNNParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: MTGNNParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.encoder: MTGNNEncoder = MTGNNEncoder(parameters, shape.input_channels, shape.input_length)
        self.projection: nn.Linear = nn.Linear(parameters.end_channels, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        features: Tensor = self.encoder(past)
        outputs: Tensor = self.projection(features.index_select(1, self.target_channels))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
