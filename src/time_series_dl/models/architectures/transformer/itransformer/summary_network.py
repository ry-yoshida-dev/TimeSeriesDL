from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import ITransformerEncoder

if TYPE_CHECKING:
    from .parameters import ITransformerParameters


class ITransformerSummaryNetwork(SummaryNetwork):
    """
    Variate tokens followed by an activation, dropout and a linear map of all flattened tokens to the output vector.

    Parameters
    ----------
    parameters : ITransformerParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: ITransformerParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: ITransformerEncoder = ITransformerEncoder(parameters, shape.input_length)
        self.activation: nn.Module = parameters.activation.build()
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.model_size * shape.input_channels, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        tokens: Tensor = self.encoder(values)
        activated: Tensor = self.activation(tokens)
        outputs: Tensor = self.projection(self.dropout(activated).flatten(start_dim=1))
        return outputs
