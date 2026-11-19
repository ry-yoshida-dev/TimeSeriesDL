from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....components.layers import TemporalLinear
from ....interfaces import ForecastNetwork, ForecastShape

if TYPE_CHECKING:
    from .parameters import NLinearParameters


class NLinearForecastNetwork(ForecastNetwork):
    """
    Linear map along time of the window relative to its last value.

    The last value is added back to the parameters in the scale of the values only.

    Parameters
    ----------
    parameters : NLinearParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: NLinearParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.linear: TemporalLinear = TemporalLinear(
            shape.input_length, shape.horizon * shape.num_parameters, shape.input_channels, parameters.is_individual
        )
        is_level: Tensor = torch.zeros(shape.num_parameters)
        is_level[list(shape.level_parameters)] = 1.0
        self.is_level: Tensor
        self.register_buffer("is_level", is_level, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        last: Tensor = past[:, -1:, :]
        mapped: Tensor = self.linear(past - last)
        outputs: Tensor = mapped.reshape(past.shape[0], self.horizon, self.num_parameters, -1).permute(0, 1, 3, 2)
        outputs = outputs + last.unsqueeze(-1) * self.is_level
        return outputs.index_select(2, self.target_channels)
