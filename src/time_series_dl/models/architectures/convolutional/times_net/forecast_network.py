from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .embedding import TimesNetEmbedding
from .times_block import TimesBlock

if TYPE_CHECKING:
    from .parameters import TimesNetParameters


class TimesNetForecastNetwork(ForecastNetwork):
    """
    Embedding of the window, a linear map along time extending it over the horizon, TimesBlocks on the extended
    sequence, and a projection of its last ``horizon`` steps to the target parameters.

    Parameters
    ----------
    parameters : TimesNetParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: TimesNetParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.embedding: TimesNetEmbedding = TimesNetEmbedding(
            shape.input_channels, parameters.model_size, shape.input_length, parameters.dropout
        )
        self.extension: nn.Linear = nn.Linear(shape.input_length, shape.input_length + shape.horizon)
        self.blocks: nn.Sequential = nn.Sequential(*(TimesBlock(parameters) for _ in range(parameters.num_layers)))
        self.projection: nn.Linear = nn.Linear(parameters.model_size, shape.output_channels * shape.num_parameters)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        tokens: Tensor = self.embedding(past)
        extended: Tensor = self.extension(tokens.transpose(1, 2)).transpose(1, 2)
        features: Tensor = self.blocks(extended)
        outputs: Tensor = self.projection(features[:, -self.horizon :])
        return outputs.reshape(-1, self.horizon, self.output_channels, self.num_parameters)
