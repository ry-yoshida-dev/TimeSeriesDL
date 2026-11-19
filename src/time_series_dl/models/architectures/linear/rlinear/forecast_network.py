from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....components.layers import TemporalLinear
from ....components.normalization import InstanceStatistics, ReversibleInstanceNormalization
from ....interfaces import ForecastNetwork, ForecastShape

if TYPE_CHECKING:
    from .parameters import RLinearParameters


class RLinearForecastNetwork(ForecastNetwork):
    """
    Reversible instance normalization of each target channel, dropout, and a linear map along time to the horizon.

    The normalization is undone on the parameters in the scale of the values only.

    Parameters
    ----------
    parameters : RLinearParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: RLinearParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        self.normalization: ReversibleInstanceNormalization = ReversibleInstanceNormalization(
            shape.output_channels, parameters.is_affine
        )
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.linear: TemporalLinear = TemporalLinear(
            shape.input_length, shape.horizon * shape.num_parameters, shape.output_channels, parameters.is_individual
        )
        is_level: Tensor = torch.zeros(shape.num_parameters, dtype=torch.bool)
        is_level[list(shape.level_parameters)] = True
        self.is_level: Tensor
        self.register_buffer("is_level", is_level, persistent=False)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        targets: Tensor = past.index_select(2, self.target_channels)
        statistics: InstanceStatistics = self.normalization.compute_statistics(targets)
        normalized: Tensor = self.dropout(self.normalization.normalize(targets, statistics))
        mapped: Tensor = self.linear(normalized)
        outputs: Tensor = mapped.reshape(past.shape[0], self.horizon, self.num_parameters, -1).transpose(2, 3)
        return torch.where(self.is_level, self.normalization.denormalize(outputs, statistics), outputs)
