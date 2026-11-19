from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import MTGNNEncoder

if TYPE_CHECKING:
    from .parameters import MTGNNParameters


class MTGNNSummaryNetwork(SummaryNetwork):
    """
    MTGNN node features followed by dropout and a linear map of all flattened node features to the output vector.

    Parameters
    ----------
    parameters : MTGNNParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: MTGNNParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: MTGNNEncoder = MTGNNEncoder(parameters, shape.input_channels, shape.input_length)
        self.dropout: nn.Dropout = nn.Dropout(parameters.dropout)
        self.projection: nn.Linear = nn.Linear(parameters.end_channels * shape.input_channels, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)
        outputs: Tensor = self.projection(self.dropout(features.flatten(start_dim=1)))
        return outputs
