from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .scale_branch import FiLMScaleBranch

if TYPE_CHECKING:
    from .parameters import FiLMParameters


class FiLMForecastNetwork(ForecastNetwork):
    """
    Each target channel is projected by a Legendre memory over windows of several multiples of the horizon, the
    memory states are filtered in frequency, the last state is reconstructed as the horizon, and a linear layer mixes
    the scales into the distribution parameters.

    Parameters
    ----------
    parameters : FiLMParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: FiLMParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.branches: nn.ModuleList = nn.ModuleList(
            FiLMScaleBranch(
                parameters.state_size, scale * shape.horizon, shape.input_length, shape.horizon, parameters.num_modes
            )
            for scale in parameters.scales
        )
        self.mixture: nn.Linear = nn.Linear(len(parameters.scales), shape.num_parameters)
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        values: Tensor = past.index_select(2, self.target_channels).transpose(1, 2)
        horizons: list[Tensor] = [branch(values) for branch in self.branches]
        mixed: Tensor = self.mixture(torch.stack(horizons, dim=-1))
        return mixed.transpose(1, 2)
