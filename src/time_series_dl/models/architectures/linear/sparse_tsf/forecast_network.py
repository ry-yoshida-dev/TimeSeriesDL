from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape

if TYPE_CHECKING:
    from .parameters import SparseTSFParameters


class SparseTSFForecastNetwork(ForecastNetwork):
    """
    Sliding aggregation of each target channel, down-sampling into periods, and a linear map without bias from the
    past periods to the future periods, shared by every phase and channel.

    The window is zero-padded at the front to whole periods, and the forecast keeps the first ``horizon`` steps of the
    whole future periods.

    Parameters
    ----------
    parameters : SparseTSFParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: SparseTSFParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.period: int = parameters.period
        self.horizon: int = shape.horizon
        self.num_parameters: int = shape.num_parameters
        num_input_periods: int = math.ceil(shape.input_length / parameters.period)
        self.num_output_periods: int = math.ceil(shape.horizon / parameters.period)
        self.padding: int = num_input_periods * parameters.period - shape.input_length
        half_period: int = parameters.period // 2
        self.aggregation: nn.Conv1d = nn.Conv1d(1, 1, 1 + 2 * half_period, padding=half_period, bias=False)
        self.linear: nn.Linear = nn.Linear(
            num_input_periods, self.num_output_periods * shape.num_parameters, bias=False
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        batch_size: int = past.shape[0]
        series: Tensor = past.index_select(2, self.target_channels).transpose(1, 2).reshape(-1, 1, past.shape[1])
        aggregated: Tensor = self.aggregation(series) + series
        padded: Tensor = functional.pad(aggregated.squeeze(1), (self.padding, 0))
        periods: Tensor = padded.reshape(padded.shape[0], -1, self.period).transpose(1, 2)
        mapped: Tensor = self.linear(periods).reshape(-1, self.period, self.num_parameters, self.num_output_periods)
        future: Tensor = mapped.permute(0, 2, 3, 1).reshape(
            -1, self.num_parameters, self.num_output_periods * self.period
        )
        outputs: Tensor = future[..., : self.horizon].reshape(batch_size, -1, self.num_parameters, self.horizon)
        return outputs.permute(0, 3, 1, 2)
