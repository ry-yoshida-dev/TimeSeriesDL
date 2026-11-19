from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import ITransformerEncoder

if TYPE_CHECKING:
    from .parameters import ITransformerParameters


class ITransformerForecastNetwork(ForecastNetwork):
    """
    Variate tokens of every input channel, of which each target channel's token is mapped linearly to its horizon.

    Parameters
    ----------
    parameters : ITransformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: ITransformerParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.encoder: ITransformerEncoder = ITransformerEncoder(parameters, shape.input_length)
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        tokens: Tensor = self.encoder(past)
        outputs: Tensor = self.projection(tokens.index_select(1, self.target_channels))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
