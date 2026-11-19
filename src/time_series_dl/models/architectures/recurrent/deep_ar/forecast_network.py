from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .recurrence import DeepARRecurrence

if TYPE_CHECKING:
    from .parameters import DeepARParameters


class DeepARForecastNetwork(ForecastNetwork):
    """
    Recurrent layers that encode the past window and then decode the horizon one step at a time.

    The past steps read every input channel. The first decoded step reads the last past target values; each later
    step reads the previous step's target values, which are the given future targets while training and the
    predicted location parameter otherwise.

    Parameters
    ----------
    parameters : DeepARParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: DeepARParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.recurrence: DeepARRecurrence = DeepARRecurrence(
            parameters, shape.input_channels, shape.output_channels, 0, shape.num_parameters, shape.location_index
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        last_targets: Tensor = past.index_select(2, self.target_channels)[:, -1:, :]
        return self.recurrence(past, last_targets, past.new_zeros(past.shape[0], self.horizon, 0), future_targets)
