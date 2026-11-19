from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import PyraformerEncoder

if TYPE_CHECKING:
    from .parameters import PyraformerParameters


class PyraformerSummaryNetwork(SummaryNetwork):
    """
    Pyraformer encoder whose features of every step are passed through GELU and dropout, flattened and mapped
    linearly to the output vector, as in the Time-Series-Library.

    Parameters
    ----------
    parameters : PyraformerParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: PyraformerParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: PyraformerEncoder = PyraformerEncoder(parameters, shape.input_length, shape.input_channels)
        self.activation: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(
            self.encoder.padded_length * self.encoder.feature_size, shape.output_size
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.dropout(self.activation(self.encoder(values)))
        outputs: Tensor = self.projection(features.flatten(start_dim=1))
        return outputs
