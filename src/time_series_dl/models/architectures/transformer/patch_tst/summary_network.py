from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor, nn

from ....interfaces import SummaryNetwork, SummaryShape
from .encoder import PatchTSTEncoder

if TYPE_CHECKING:
    from .parameters import PatchTSTParameters


class PatchTSTSummaryNetwork(SummaryNetwork):
    """
    Patch encoder of each channel followed by a linear map of the features of all channels to the output vector.

    Parameters
    ----------
    parameters : PatchTSTParameters
        The model configuration.
    shape : SummaryShape
        The input and output dimensions.
    """

    def __init__(self, parameters: PatchTSTParameters, shape: SummaryShape) -> None:
        super().__init__()
        self.encoder: PatchTSTEncoder = PatchTSTEncoder(parameters, shape.input_length)
        self.dropout: nn.Dropout = nn.Dropout(parameters.head_dropout)
        self.projection: nn.Linear = nn.Linear(self.encoder.feature_size * shape.input_channels, shape.output_size)

    @override
    def forward(self, values: Tensor) -> Tensor:
        features: Tensor = self.encoder(values)
        outputs: Tensor = self.projection(self.dropout(features.flatten(start_dim=1)))
        return outputs
