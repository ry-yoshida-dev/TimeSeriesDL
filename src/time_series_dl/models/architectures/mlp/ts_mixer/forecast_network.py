from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape

if TYPE_CHECKING:
    from .parameters import TSMixerParameters


class TSMixerForecastNetwork(ForecastNetwork):
    """
    Mixer blocks over every input channel, then a linear map along time of each channel to the horizon.

    Every channel is forecast, and the target channels are selected from the result.

    Parameters
    ----------
    parameters : TSMixerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: TSMixerParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.mixer: nn.Sequential = parameters.build_mixer(shape.input_length, shape.input_channels)
        self.projection: nn.Linear = nn.Linear(shape.input_length, shape.horizon * shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        mixed: Tensor = self.mixer(past)
        projected: Tensor = self.projection(mixed.transpose(1, 2))
        outputs: Tensor = projected.reshape(past.shape[0], -1, self.horizon, self.num_parameters).transpose(1, 2)
        return outputs.index_select(2, self.target_channels)
