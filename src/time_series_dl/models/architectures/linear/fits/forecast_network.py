from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from ....interfaces import ForecastNetwork, ForecastShape
from .frequency_interpolation import FITSFrequencyInterpolation

if TYPE_CHECKING:
    from .parameters import FITSParameters


class FITSForecastNetwork(ForecastNetwork):
    """
    Frequency interpolation of each target channel to the window extended by the horizon, of which the last
    ``horizon`` steps are the forecast; each distribution parameter has its own output frequencies.

    Parameters
    ----------
    parameters : FITSParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: FITSParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.interpolation: FITSFrequencyInterpolation = FITSFrequencyInterpolation(
            shape.input_length,
            shape.input_length + shape.horizon,
            parameters.compute_cutoff_frequency(shape.input_length),
            shape.output_channels,
            parameters.is_individual,
            shape.num_parameters,
        )
        self.target_channels: Tensor
        self.register_buffer("target_channels", torch.tensor(shape.target_channels), persistent=False)

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        extended: Tensor = self.interpolation(past.index_select(2, self.target_channels))
        return extended[:, -self.horizon :]
