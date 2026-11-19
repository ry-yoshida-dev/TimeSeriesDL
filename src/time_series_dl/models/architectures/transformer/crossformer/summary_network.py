from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import CrossformerEncoder

if TYPE_CHECKING:
    from .parameters import CrossformerParameters


class CrossformerSummaryNetwork(SummaryNetwork):
    """
    Crossformer encoder whose coarsest-scale tokens of every channel are flattened and mapped linearly to the output
    vector after dropout, as in the Time-Series-Library.

    Parameters
    ----------
    parameters : CrossformerParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: CrossformerParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: CrossformerEncoder = CrossformerEncoder(parameters, shape.input_length, shape.input_channels)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(
            shape.input_channels * self.encoder.scale_segments[-1] * parameters.model_size, shape.output_size
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: Tensor = self.encoder(values)[-1]
        outputs: Tensor = self.projection(self.dropout(tokens.flatten(start_dim=1)))
        return outputs
