from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .block import StemGNNBlock
from .latent_correlation_layer import StemGNNLatentCorrelationLayer

if TYPE_CHECKING:
    from .parameters import StemGNNParameters


class StemGNNForecastNetwork(ForecastNetwork):
    """
    StemGNN over every input channel as a node: a latent correlation layer infers the graph from the window, StemGNN
    blocks each forecast from the backcast of the previous one, and an output layer maps the summed forecast features
    of each target node to its whole horizon at once.

    Parameters
    ----------
    parameters : StemGNNParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: StemGNNParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.correlation: StemGNNLatentCorrelationLayer = StemGNNLatentCorrelationLayer(parameters)
        self.blocks: nn.ModuleList = nn.ModuleList(
            StemGNNBlock(parameters, shape.input_length, is_backcasting=index < parameters.num_blocks - 1)
            for index in range(parameters.num_blocks)
        )
        self.output: nn.Sequential = nn.Sequential(
            nn.Linear(shape.input_length, shape.input_length),
            nn.LeakyReLU(),
            nn.Linear(shape.input_length, shape.horizon * shape.num_parameters),
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        values: Tensor = past.transpose(1, 2)
        polynomials: Tensor = self.correlation(values)
        forecasts: list[Tensor] = []
        for block in self.blocks:
            forecast: Tensor
            backcast: Tensor | None
            forecast, backcast = block(values, polynomials)
            forecasts.append(forecast)
            if backcast is not None:
                values = backcast
        summed: Tensor = torch.stack(forecasts).sum(dim=0)
        outputs: Tensor = self.output(summed.index_select(1, self.target_channels))
        return outputs.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
