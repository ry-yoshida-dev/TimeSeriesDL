from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....components.layers import MultilayerPerceptron
from ....interfaces import ForecastNetwork, ForecastShape

if TYPE_CHECKING:
    from .parameters import MLPParameters


class MLPForecastNetwork(ForecastNetwork):
    """
    Multilayer perceptron from the flattened past window to the flattened horizon.

    Parameters
    ----------
    parameters : MLPParameters
        The layer configuration.
    shape : ForecastShape
        The dimensions of the forecast.
    """

    def __init__(self, parameters: MLPParameters, shape: ForecastShape) -> None:
        super().__init__()
        self.horizon: int = shape.horizon
        self.output_channels: int = shape.output_channels
        self.num_parameters: int = shape.num_parameters
        self.layers: MultilayerPerceptron = MultilayerPerceptron(
            shape.input_length * shape.input_channels,
            parameters.hidden_sizes,
            shape.horizon * shape.output_channels * shape.num_parameters,
            parameters.activation,
            parameters.dropout,
        )

    @override
    def forward(self, past: Tensor, future_targets: Tensor | None = None) -> Tensor:
        outputs: Tensor = self.layers(past.flatten(start_dim=1))
        return outputs.reshape(-1, self.horizon, self.output_channels, self.num_parameters)
