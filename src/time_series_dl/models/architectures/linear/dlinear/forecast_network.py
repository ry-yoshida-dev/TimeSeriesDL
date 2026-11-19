from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .decomposition_linear import DecompositionLinear

if TYPE_CHECKING:
    from .parameters import DLinearParameters


class DLinearForecastNetwork(ForecastNetwork):
    """
    Decomposition into trend and seasonal parts, each mapped linearly to the horizon of every channel.

    Parameters
    ----------
    parameters : DLinearParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: DLinearParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.linear: DecompositionLinear = DecompositionLinear(
            parameters.kernel_size,
            shape.input_length,
            shape.horizon * shape.num_parameters,
            shape.input_channels,
            parameters.is_individual,
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        mapped: Tensor = self.linear(past)
        outputs: Tensor = mapped.reshape(past.shape[0], self.horizon, self.num_parameters, -1).permute(0, 1, 3, 2)
        return outputs.index_select(2, self.target_channels)
