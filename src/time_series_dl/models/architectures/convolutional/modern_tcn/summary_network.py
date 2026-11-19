from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import ModernTCNEncoder

if TYPE_CHECKING:
    from .parameters import ModernTCNParameters


class ModernTCNSummaryNetwork(SummaryNetwork):
    """
    ModernTCN backbone followed by GELU, dropout and a linear map of the features of every variable and patch to the
    output vector.

    Parameters
    ----------
    parameters : ModernTCNParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: ModernTCNParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: ModernTCNEncoder = ModernTCNEncoder(parameters, shape.input_length, shape.input_channels)
        self.activation: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.projection: nn.Linear = nn.Linear(
            shape.input_channels * self.encoder.output_size * self.encoder.output_length, shape.output_size
        )

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.dropout(self.activation(self.encoder(values)))
        outputs: Tensor = self.projection(features.flatten(start_dim=1))
        return outputs
