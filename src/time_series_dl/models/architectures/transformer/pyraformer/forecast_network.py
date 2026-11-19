from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import ForecastNetwork, ForecastShape
from .encoder import PyraformerEncoder

if TYPE_CHECKING:
    from .parameters import PyraformerParameters


class PyraformerForecastNetwork(ForecastNetwork):
    """
    Pyraformer encoder whose features of the last input step, concatenated over all scales, are mapped linearly to the
    whole horizon of every target channel.

    Parameters
    ----------
    parameters : PyraformerParameters
        The model configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: PyraformerParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.encoder: PyraformerEncoder = PyraformerEncoder(parameters, shape.input_length, shape.input_channels)
        self.projection: nn.Linear = nn.Linear(
            self.encoder.feature_size, shape.horizon * shape.output_channels * shape.num_parameters
        )

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        features: Tensor = self.encoder(past)[:, -1]
        outputs: Tensor = self.projection(features)
        return outputs.reshape(-1, self.horizon, self.output_channels, self.num_parameters)
