from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

from ....components.layers import MultilayerPerceptron
from ....interfaces import SummaryNetwork, SummaryShape

if TYPE_CHECKING:
    from .parameters import MLPParameters


class MLPSummaryNetwork(SummaryNetwork):
    """
    Multilayer perceptron from the flattened sequence to the output vector.

    Parameters
    ----------
    parameters : MLPParameters
        The layer configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: MLPParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.layers: MultilayerPerceptron = MultilayerPerceptron(
            shape.input_length * shape.input_channels,
            parameters.hidden_sizes,
            shape.output_size,
            parameters.activation,
            parameters.dropout,
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        outputs: Tensor = self.layers(values.flatten(start_dim=1))
        return outputs
